"""FX graph passes for the torchlite compiler.

Every transformation after trace() is an FX graph pass with the signature
(gm, example_inputs, **kwargs) -> PassResult. This module contains all
passes that transform the graph, from initial verification through
decomposition, fusion, and code generation.
"""
import logging
import operator
import warnings
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule
from torch.overrides import resolve_name

from torch._torchlite.collectives import (
    _allgather,
    _allreduce,
    _BINARY_ELEMENTWISE_OPS,
    _has_dtensor_params,
    _is_replicate,
    _is_shard,
    _MATMUL_OPS,
    _NORM_OPS,
    _POINTWISE_PLACEMENT_OPS,
    _propagate_binary,
    _propagate_embedding,
    _propagate_matmul,
    _propagate_norm,
    _propagate_pointwise,
    _propagate_reduction,
    _propagate_view,
    _REDUCTION_OPS,
    _reduce_scatter,
    _VIEW_OPS,
)
from torch._torchlite.ops import (
    _save_for_backward,
    _save_rng_state,
    _load_rng_state,
    adamw_step,
    param_update,
)

log = logging.getLogger(__name__)


@dataclass
class PassResult:
    gm: GraphModule


_graph_meta_store: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _graph_meta(graph):
    meta = _graph_meta_store.get(graph)
    if meta is None:
        meta = {}
        _graph_meta_store[graph] = meta
    return meta


@dataclass
class FusionGroup:
    group_id: int
    node_names: List[str]
    op_names: List[str]
    shape: Optional[List[int]]
    inputs: List[str]
    output: str


@dataclass
class FusedOp:
    op_name: str
    args: List  # each is ("input", idx), ("tmp", idx), or ("const", value)


# eq=False: FusedKernel instances are used as FX node targets and must
# be compared by identity, not by value. Two kernels with the same ops
# but different graph positions are distinct targets.
@dataclass(eq=False)
class FusedKernel:
    name: str
    ops: List[FusedOp]
    n_inputs: int
    shape: Optional[List[int]] = None

    def __post_init__(self):
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.name}: fused kernel placeholder — run triton_codegen "
            f"and precompile passes, then load the generated module"
        )


_VARARGS_TENSOR_METHODS = frozenset({
    "reshape", "view", "expand", "repeat", "permute",
    "flip", "squeeze",
})

_DUNDER_TO_OP = {
    "__add__": "add",
    "__sub__": "sub",
    "__mul__": "mul",
    "__matmul__": "matmul",
    "__truediv__": "div",
    "__floordiv__": "floor_divide",
    "__mod__": "remainder",
    "__pow__": "pow",
    "__neg__": "neg",
    "__abs__": "abs",
    "__eq__": "eq",
    "__ne__": "ne",
    "__lt__": "lt",
    "__le__": "le",
    "__gt__": "gt",
    "__ge__": "ge",
    "__invert__": "bitwise_not",
    "__and__": "bitwise_and",
    "__or__": "bitwise_or",
    "__xor__": "bitwise_xor",
}

_REVERSE_DUNDERS = {
    "__radd__", "__rsub__", "__rmul__", "__rmatmul__",
    "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__",
    "__rand__", "__ror__", "__rxor__",
}

_DUNDER_INPLACE = {
    "__iadd__": "add_",
    "__isub__": "sub_",
    "__imul__": "mul_",
    "__itruediv__": "div_",
    "__iand__": "bitwise_and_",
    "__ior__": "bitwise_or_",
    "__ixor__": "bitwise_xor_",
}


def _normalize_target(func):
    name = getattr(func, "__name__", None)

    if name == "__getitem__":
        return operator.getitem

    resolved = resolve_name(func)
    if resolved is not None:
        dot = resolved.rfind(".")
        if dot >= 0:
            namespace, attr = resolved[:dot], resolved[dot + 1:]

            if attr in _REVERSE_DUNDERS:
                return func
            if attr in _DUNDER_TO_OP:
                return getattr(torch, _DUNDER_TO_OP[attr])
            if attr in _DUNDER_INPLACE:
                return getattr(torch.Tensor, _DUNDER_INPLACE[attr])

            if namespace == "torch.Tensor" and not attr.startswith("_"):
                if attr in _VARARGS_TENSOR_METHODS:
                    return func
                torch_func = getattr(torch, attr, None)
                if torch_func is not None and callable(torch_func):
                    return torch_func

        return func

    if name is not None:
        torch_func = getattr(torch, name, None)
        if torch_func is not None and callable(torch_func):
            return torch_func

    return func


def _set_dtensor_meta(tensor, node):
    if hasattr(tensor, "placements") and hasattr(tensor, "device_mesh"):
        p = tensor.placements[0]
        if hasattr(p, "dim"):
            node.meta["dtensor_spec"] = ("Shard", p.dim)
        else:
            node.meta["dtensor_spec"] = ("Replicate", None)


def normalize(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    graph = gm.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        new_target = _normalize_target(node.target)
        if new_target is not node.target:
            node.target = new_target
            name = getattr(new_target, "__name__", str(new_target))
            node.name = graph._graph_namespace.create_name(name, None)
    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def annotate_dtensor(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    ph_idx = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            param_val = _deep_getattr(gm, node.target)
            _set_dtensor_meta(param_val, node)
        elif node.op == "placeholder":
            if ph_idx < len(example_inputs):
                _set_dtensor_meta(example_inputs[ph_idx], node)
            ph_idx += 1
    return PassResult(gm=gm)


def verify_graph(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target

        if not callable(target):
            raise ValueError(
                f"Node '{node.name}': target is not callable: {target!r}"
            )

        if not hasattr(target, "__name__"):
            raise ValueError(
                f"Node '{node.name}': target has no __name__: {target!r}"
            )

        # Detect torch.Tensor methods that should have been normalized to
        # their torch.* functional equivalent during tracing. If resolve_name
        # identifies a Tensor method and the corresponding torch.* function
        # exists, normalization missed it. Varargs methods (reshape, view, etc.)
        # are excluded because their torch.* equivalents take a tuple instead.
        resolved = resolve_name(target)
        if resolved is not None and resolved.startswith("torch.Tensor."):
            attr = resolved.rsplit(".", 1)[1]
            if not attr.startswith("_") and attr not in _VARARGS_TENSOR_METHODS:
                torch_func = getattr(torch, attr, None)
                if torch_func is not None and callable(torch_func):
                    raise ValueError(
                        f"Node '{node.name}': target {resolved} should have "
                        f"been normalized to torch.{attr}"
                    )
    return PassResult(gm=gm)


def _find_functional_variant(name):
    """Find the functional (non-in-place) variant of an in-place op name.

    Handles simple cases (add_ → torch.add) and compound names
    (addcmul_ → torch.addcmul, scatter_add_ → torch.Tensor.scatter_add).
    Returns None if no functional variant is found.
    """
    base = name[:-1]
    functional = getattr(torch, base, None)
    if functional is not None and callable(functional):
        return functional
    functional = getattr(torch.Tensor, base, None)
    if functional is not None and callable(functional):
        return functional
    return None


def functionalize(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    graph = gm.graph
    mutations = []
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if not name.endswith("_") or name.startswith("__"):
            continue
        functional = _find_functional_variant(name)
        if functional is None:
            continue

        # The first arg is the tensor mutated in-place. After switching to
        # the functional variant, all subsequent uses of that tensor must
        # reference the output of the functional op instead — otherwise
        # they would silently read the stale, pre-mutation value.
        mutated = node.args[0]
        if isinstance(mutated, torch.fx.Node):
            mutations.append((mutated, node))
            node_idx = node_order[node]
            for user in list(mutated.users):
                # Preserve topological order and semantics: only redirect
                # users that occur after the mutation. Earlier users must
                # continue to read the pre-mutation value.
                user_idx = node_order.get(user, -1)
                if user is not node and user_idx > node_idx:
                    user.replace_input_with(mutated, node)

            # When the mutated tensor is a slice (getitem) of a parent,
            # the original in-place op also mutated the parent through
            # the view aliasing. We must write the functional result back
            # into the parent so downstream ops that read the full parent
            # tensor (e.g. view(parent, ...)) see the updated values.
            if (
                mutated.op == "call_function"
                and mutated.target is operator.getitem
                and len(mutated.args) >= 2
            ):
                parent = mutated.args[0]
                key = mutated.args[1]
                with graph.inserting_after(node):
                    setitem_node = graph.call_function(
                        operator.setitem, (parent, key, node)
                    )
                    setitem_node.name = graph._graph_namespace.create_name(
                        "setitem_back", None
                    )

        node.target = functional
        node.name = graph._graph_namespace.create_name(name[:-1], None)

    # Insert copy-back nodes: the graph body is now purely functional
    # (fusible), but we still need to reflect each mutation back into the
    # original tensor so external observers see the correct state.
    # Walk mutations to find each base tensor's final value. When a
    # tensor is mutated multiple times (a.add_(b); a.add_(c)), the
    # chain is: base=a → first_result → second_result. We only need
    # one copy_(base, last_result) per base.
    if mutations:
        output_node = None
        for n in graph.nodes:
            if n.op == "output":
                output_node = n
                break
        graph.inserting_before(output_node)
        node_to_base = {}
        final_for_base = {}
        for mutated, result in mutations:
            base = node_to_base.get(mutated, mutated)
            node_to_base[result] = base
            final_for_base[base] = result
        for base, final_value in final_for_base.items():
            copy_node = graph.call_function(
                torch.Tensor.copy_, (base, final_value)
            )
            copy_node.name = graph._graph_namespace.create_name("copy_back", None)
            _set_phase(copy_node, "copy-back")

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def _set_phase(node, phase):
    node.meta["phase"] = phase


def _emit_sgd_update(graph, param_node, grad_node, short, lr):
    scaled_grad = graph.call_function(torch.mul, (lr, grad_node))
    scaled_grad.name = graph._graph_namespace.create_name(
        short + "_step", None
    )
    _set_phase(scaled_grad, "optimizer")

    new_param = graph.call_function(torch.sub, (param_node, scaled_grad))
    new_param.name = graph._graph_namespace.create_name(
        short + "_new", None
    )
    _set_phase(new_param, "optimizer")

    copy_node = graph.call_function(
        param_update, (param_node, new_param)
    )
    copy_node.name = graph._graph_namespace.create_name(
        short + "_update", None
    )
    _set_phase(copy_node, "optimizer")


def optimizer(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> PassResult:
    graph = gm.graph
    param_grad_info = _graph_meta(gm.graph).get("param_grad_info", {})
    if not param_grad_info:
        return PassResult(gm=gm)

    output_node = None
    for n in graph.nodes:
        if n.op == "output":
            output_node = n
            break

    orig_output = output_node.args[0]
    if not isinstance(orig_output, (tuple, list)):
        return PassResult(gm=gm)

    param_nodes = {}
    for n in graph.nodes:
        if n.op == "get_attr":
            param_nodes[n.target] = n

    graph.inserting_before(output_node)

    if optimizer_type == "adamw":
        adam_state = {}
        step_tensor = torch.tensor(0, dtype=torch.long)
        step_target = "_adam_step"
        gm.register_buffer(step_target, step_tensor, persistent=True)
        step_node = graph.get_attr(step_target)
        _set_phase(step_node, "optimizer")

        for param_name in param_grad_info:
            param_val = _deep_getattr(gm, param_name)
            m = torch.zeros_like(param_val)
            v = torch.zeros_like(param_val)
            m_target = f"_adam_m_{param_name.replace('.', '_')}"
            v_target = f"_adam_v_{param_name.replace('.', '_')}"
            gm.register_buffer(m_target, m, persistent=True)
            gm.register_buffer(v_target, v, persistent=True)

            m_node = graph.get_attr(m_target)
            _set_phase(m_node, "optimizer")
            v_node = graph.get_attr(v_target)
            _set_phase(v_node, "optimizer")
            adam_state[param_name] = (m_node, v_node)

        step_inc = graph.call_function(torch.add, (step_node, 1))
        step_inc.name = graph._graph_namespace.create_name("adam_step_inc", None)
        _set_phase(step_inc, "optimizer")
        step_copy = graph.call_function(torch.Tensor.copy_, (step_node, step_inc))
        step_copy.name = graph._graph_namespace.create_name("adam_step_update", None)
        _set_phase(step_copy, "optimizer")

        for param_name, grad_idx in param_grad_info.items():
            param_node = param_nodes[param_name]
            grad_node = orig_output[1 + grad_idx]
            m_node, v_node = adam_state[param_name]
            short = param_name.split(".")[-1]

            update_node = graph.call_function(
                adamw_step,
                (
                    param_node, grad_node,
                    m_node, v_node, step_inc,
                    lr, betas[0], betas[1], eps, weight_decay,
                ),
            )
            update_node.name = graph._graph_namespace.create_name(
                short + "_adam_update", None
            )
            _set_phase(update_node, "optimizer")

    else:
        for param_name, grad_idx in param_grad_info.items():
            param_node = param_nodes[param_name]
            grad_node = orig_output[1 + grad_idx]
            short = param_name.split(".")[-1]
            _emit_sgd_update(graph, param_node, grad_node, short, lr)

    loss_node = orig_output[0]
    output_node.args = (loss_node,)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def _deep_getattr(obj, target):
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _deep_setattr(obj, target, value):
    parts = target.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            setattr(obj, part, torch.nn.Module())
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


_PROVENANCE_KEYS = frozenset({"phase", "bwd_of", "dtensor_spec", "rng_replay_for"})


def _is_torch_op(target):
    if target is operator.getitem:
        return False
    if isinstance(target, FusedKernel):
        return False
    if isinstance(target, torch._ops.OpOverload):
        return True
    module = getattr(target, "__module__", "") or ""
    if "torchlite" in module:
        return False
    return module.startswith("torch")


class _DecompRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Intercepts aten ops during decomposition, applies decompositions from
    the table for non-core ops, and records leaf (core) ops directly into an
    existing FX graph. This replaces make_fx with inline graph surgery.
    """

    def __init__(self, graph, id_to_node, decomp_table, new_nodes, prov):
        self.graph = graph
        self.id_to_node = id_to_node
        self.decomp_table = decomp_table
        self.new_nodes = new_nodes
        self.prov = prov

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        decomp = self.decomp_table.get(func)
        if decomp is not None:
            return decomp(*args, **kwargs)

        result = func(*args, **kwargs)

        def _to_fx(x):
            if isinstance(x, torch.Tensor) and id(x) in self.id_to_node:
                return self.id_to_node[id(x)]
            if isinstance(x, (list, tuple)):
                return type(x)(_to_fx(i) for i in x)
            return x

        fx_args = tuple(
            _to_fx(a) if not isinstance(a, (list, tuple))
            else type(a)(_to_fx(x) for x in a)
            for a in args
        )
        fx_kwargs = {k: _to_fx(v) for k, v in kwargs.items()}

        new_node = self.graph.call_function(func, fx_args, fx_kwargs)
        pkt = getattr(func, "overloadpacket", None)
        name_hint = getattr(pkt, "__name__", None) if pkt else None
        if name_hint is None:
            name_hint = getattr(func, "__name__", "decomp")
        new_node.name = self.graph._graph_namespace.create_name(name_hint, None)

        if isinstance(result, torch.Tensor):
            new_node.meta["shape"] = list(result.shape)
            new_node.meta["dtype"] = result.dtype
            self.id_to_node[id(result)] = new_node
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    get_node = self.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    get_node.name = self.graph._graph_namespace.create_name(
                        f"{name_hint}_item", None
                    )
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    self.id_to_node[id(r)] = get_node
                    self.new_nodes.append(get_node)

        for k, v in self.prov.items():
            new_node.meta[k] = v

        self.new_nodes.append(new_node)
        return result


def _storage_key(t):
    return (t.data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))


class _BackwardRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Intercepts ATen ops during torch.autograd.grad and records them into
    the FX graph with phase="backward". Modeled on _DecompRecorder but with
    no decomposition table — every ATen op is recorded directly.

    Uses id(), storage-based keys, AND data_ptr-based view reconstruction
    to match tensors to FX nodes. Autograd's C++ backward functions may
    create views (reshape, transpose) of saved tensors that never pass
    through Python dispatch — we detect these by data_ptr and insert
    aten.as_strided nodes to reconstruct them.
    """

    def __init__(self, graph, id_to_node, storage_to_node,
                 dptr_to_node, live_tensors):
        self.graph = graph
        self.id_to_node = id_to_node
        self.storage_to_node = storage_to_node
        self.dptr_to_node = dptr_to_node
        self.live_tensors = live_tensors

    def _lookup(self, x):
        node = self.id_to_node.get(id(x))
        if node is not None:
            return node
        node = self.storage_to_node.get(_storage_key(x))
        if node is not None:
            return node
        # The tensor might be a view created inside a C++ backward function
        # (e.g. reshape or transpose) that was never dispatched through
        # Python. Reconstruct it via aten.as_strided from the original.
        entry = self.dptr_to_node.get(x.data_ptr())
        if entry is not None:
            src_node = entry
            view_node = self.graph.call_function(
                torch.ops.aten.as_strided.default,
                (src_node, list(x.shape), list(x.stride()), x.storage_offset()),
            )
            view_node.name = self.graph._graph_namespace.create_name(
                "bwd_view", None
            )
            view_node.meta["shape"] = list(x.shape)
            view_node.meta["dtype"] = x.dtype
            _set_phase(view_node, "backward")
            self._track(x, view_node)
            return view_node
        return None

    def _track(self, tensor, node):
        self.id_to_node[id(tensor)] = node
        self.storage_to_node[_storage_key(tensor)] = node
        if tensor.data_ptr() not in self.dptr_to_node:
            self.dptr_to_node[tensor.data_ptr()] = node
        self.live_tensors.append(tensor)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        def _to_fx(x):
            if isinstance(x, torch.Tensor):
                node = self._lookup(x)
                if node is not None:
                    return node
            if isinstance(x, (list, tuple)):
                return type(x)(_to_fx(i) for i in x)
            return x

        fx_args = tuple(
            _to_fx(a) if not isinstance(a, (list, tuple))
            else type(a)(_to_fx(x) for x in a)
            for a in args
        )
        fx_kwargs = {k: _to_fx(v) for k, v in kwargs.items()}

        new_node = self.graph.call_function(func, fx_args, fx_kwargs)
        pkt = getattr(func, "overloadpacket", None)
        name_hint = getattr(pkt, "__name__", None) if pkt else None
        if name_hint is None:
            name_hint = getattr(func, "__name__", "bwd")
        new_node.name = self.graph._graph_namespace.create_name(name_hint, None)
        _set_phase(new_node, "backward")

        if isinstance(result, torch.Tensor):
            new_node.meta["shape"] = list(result.shape)
            new_node.meta["dtype"] = result.dtype
            self._track(result, new_node)
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    get_node = self.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    get_node.name = self.graph._graph_namespace.create_name(
                        f"{name_hint}_item", None
                    )
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    _set_phase(get_node, "backward")
                    self._track(r, get_node)

        return result


class _ForwardDecomposer(torch.utils._python_dispatch.TorchDispatchMode):
    """Intercepts ATen ops inside compound forward ops (like rms_norm,
    layer_norm) and records them as FX nodes. This makes internal
    intermediates (saved by autograd for backward) visible in the graph.
    Only activates for ops that dispatch to multiple ATen ops — single-op
    forwards pass through with an empty decomp_nodes list.
    """

    def __init__(self, graph, id_to_node, storage_to_node,
                 decomp_nodes, live_tensors, prov):
        self.graph = graph
        self.id_to_node = id_to_node
        self.storage_to_node = storage_to_node
        self.decomp_nodes = decomp_nodes
        self.live_tensors = live_tensors
        self.prov = prov

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        def _to_fx(x):
            if isinstance(x, torch.Tensor):
                node = self.id_to_node.get(id(x))
                if node is not None:
                    return node
                node = self.storage_to_node.get(_storage_key(x))
                if node is not None:
                    return node
            if isinstance(x, (list, tuple)):
                return type(x)(_to_fx(i) for i in x)
            return x

        fx_args = tuple(
            _to_fx(a) if not isinstance(a, (list, tuple))
            else type(a)(_to_fx(x) for x in a)
            for a in args
        )
        fx_kwargs = {k: _to_fx(v) for k, v in kwargs.items()}

        new_node = self.graph.call_function(func, fx_args, fx_kwargs)
        pkt = getattr(func, "overloadpacket", None)
        name_hint = getattr(pkt, "__name__", None) if pkt else None
        if name_hint is None:
            name_hint = getattr(func, "__name__", "fwd_decomp")
        new_node.name = self.graph._graph_namespace.create_name(name_hint, None)
        _set_phase(new_node, "forward")

        if isinstance(result, torch.Tensor):
            new_node.meta["shape"] = list(result.shape)
            new_node.meta["dtype"] = result.dtype
            self.id_to_node[id(result)] = new_node
            self.storage_to_node[_storage_key(result)] = new_node
            self.live_tensors.append(result)
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    get_node = self.graph.call_function(
                        operator.getitem, (new_node, i)
                    )
                    get_node.name = self.graph._graph_namespace.create_name(
                        f"{name_hint}_item", None
                    )
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    _set_phase(get_node, "forward")
                    self.id_to_node[id(r)] = get_node
                    self.storage_to_node[_storage_key(r)] = get_node
                    self.live_tensors.append(r)
                    self.decomp_nodes.append(get_node)

        self.decomp_nodes.append(new_node)
        return result


def decompose(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    from torch._decomp import core_aten_decompositions

    graph = gm.graph
    decomp_table = core_aten_decompositions()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if not _is_torch_op(node.target):
            continue

        tensor_map = {}
        id_to_node = {}
        can_decompose = True

        for a in _iter_node_args(node):
            if not isinstance(a, torch.fx.Node):
                continue
            if a in tensor_map:
                continue
            shape = a.meta.get("shape")
            if shape is None:
                can_decompose = False
                break
            dtype = a.meta.get("dtype", torch.float32)
            t = torch.empty(shape if shape else [], dtype=dtype, device="meta")
            tensor_map[a] = t
            id_to_node[id(t)] = a

        if not can_decompose or not tensor_map:
            continue

        prov = {k: node.meta[k] for k in _PROVENANCE_KEYS if k in node.meta}
        new_nodes: list = []

        def _resolve(x):
            if isinstance(x, torch.fx.Node):
                return tensor_map[x]
            if isinstance(x, (list, tuple)):
                return type(x)(_resolve(i) for i in x)
            return x

        real_args = tuple(_resolve(a) for a in node.args)
        real_kwargs = {k: _resolve(v) for k, v in node.kwargs.items()}

        graph.inserting_before(node)
        try:
            with _DecompRecorder(graph, id_to_node, decomp_table, new_nodes, prov):
                node.target(*real_args, **real_kwargs)
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as e:
            op_name = getattr(node.target, "__name__", str(node.target))
            log.debug("decompose: skipping %s (%s)", op_name, e)
            for n in reversed(new_nodes):
                if not n.users:
                    graph.erase_node(n)
            continue

        if not new_nodes:
            continue

        if len(new_nodes) == 1 and new_nodes[0].target is node.target:
            graph.erase_node(new_nodes[0])
            continue

        result_node = new_nodes[-1]
        node.replace_all_uses_with(result_node)
        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def _iter_node_args(node):
    for a in node.args:
        if isinstance(a, (list, tuple)):
            yield from a
        else:
            yield a
    for v in node.kwargs.values():
        if isinstance(v, (list, tuple)):
            yield from v
        else:
            yield v


def rng_functionalize(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    graph = gm.graph

    fwd_dropout_nodes = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if getattr(node.target, "__name__", "") != "dropout":
            continue
        if node.meta.get("phase", "forward") == "forward":
            fwd_dropout_nodes.append(node)

    if not fwd_dropout_nodes:
        return PassResult(gm=gm)

    # Check if any backward node references a forward dropout via
    # rng_replay_for. If not (e.g. dispatcher-based autograd where the
    # autograd tape handles dropout masks internally), skip inserting
    # save/load_rng_state to avoid dead code that breaks CUDA graph capture.
    has_rng_replay = False
    for node in graph.nodes:
        if node.op == "call_function" and node.meta.get("rng_replay_for") is not None:
            has_rng_replay = True
            break
    if not has_rng_replay:
        return PassResult(gm=gm)

    state_map = {}
    for dropout_node in fwd_dropout_nodes:
        graph.inserting_before(dropout_node)
        state_node = graph.call_function(_save_rng_state, ())
        state_node.name = graph._graph_namespace.create_name("rng_state", None)
        _set_phase(state_node, "forward")
        state_map[dropout_node] = state_node

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        fwd_ref = node.meta.get("rng_replay_for")
        if fwd_ref is None or fwd_ref not in state_map:
            continue
        state_node = state_map[fwd_ref]
        graph.inserting_before(node)
        restore = graph.call_function(_load_rng_state, (state_node,))
        restore.name = graph._graph_namespace.create_name("restore_rng", None)
        _set_phase(restore, "backward")
        restore.meta["bwd_of"] = node.meta.get("bwd_of")

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def min_cut_partition(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    # Despite the name, this does not implement a true min-cut. It saves *all*
    # forward activations consumed by backward nodes by inserting explicit
    # save_for_backward identity nodes at the forward/backward boundary.
    graph = gm.graph

    bwd_nodes = []
    for n in graph.nodes:
        if n.op == "call_function" and n.meta.get("phase") == "backward":
            bwd_nodes.append(n)

    if not bwd_nodes:
        return PassResult(gm=gm)

    # Find forward call_function nodes (with tensor shape) that backward reads.
    # Parameters (get_attr) and inputs (placeholder) are always available
    # and don't count as "saved activations" — only computed intermediates do.
    saved_fwd = set()
    for bwd_node in bwd_nodes:
        for arg in _iter_node_args(bwd_node):
            if not isinstance(arg, torch.fx.Node):
                continue
            if arg.op != "call_function":
                continue
            if arg.meta.get("phase", "forward") != "forward":
                continue
            if "shape" not in arg.meta:
                continue
            saved_fwd.add(arg)

    if not saved_fwd:
        return PassResult(gm=gm)

    fwd_to_save = [n for n in graph.nodes if n in saved_fwd]
    first_bwd = bwd_nodes[0]

    save_map = {}
    graph.inserting_before(first_bwd)
    for fwd_node in fwd_to_save:
        save_node = graph.call_function(_save_for_backward, (fwd_node,))
        save_node.name = graph._graph_namespace.create_name(
            f"save_{fwd_node.name}", None
        )
        _set_phase(save_node, "save")
        if "shape" in fwd_node.meta:
            save_node.meta["shape"] = fwd_node.meta["shape"]
        save_map[fwd_node] = save_node

    for n in list(graph.nodes):
        if n.meta.get("phase") != "backward":
            continue
        for original, save_node in save_map.items():
            n.replace_input_with(original, save_node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def activation_checkpoint(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    ops_to_recompute: Optional[List[str]] = None,
) -> PassResult:
    graph = gm.graph

    save_nodes = {}
    save_by_name = {}
    for n in graph.nodes:
        if n.op == "call_function" and n.target is _save_for_backward:
            fwd_node = n.args[0]
            save_nodes[fwd_node] = n
            save_by_name[fwd_node.name] = (n, fwd_node)

    if not save_nodes:
        return PassResult(gm=gm)

    cheap = {"sin", "cos", "add", "sub", "neg", "mul"}
    if ops_to_recompute is None:
        ops_to_recompute = []
        for fwd_name, (save_node, fwd_node) in save_by_name.items():
            if fwd_node.op != "call_function":
                continue
            op_name = _aten_op_name(fwd_node.target)
            if op_name not in cheap:
                continue
            can_recompute = True
            for arg in fwd_node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                if arg.op in ("get_attr", "placeholder"):
                    continue
                if arg.meta.get("phase", "forward") != "forward":
                    continue
                if arg not in save_nodes:
                    can_recompute = False
                    break
            if can_recompute:
                ops_to_recompute.append(fwd_name)

    if not ops_to_recompute:
        return PassResult(gm=gm)

    ops_set = set(ops_to_recompute)
    ordered = []
    for n in graph.nodes:
        if n.name in ops_set and n.name in save_by_name:
            ordered.append(n.name)

    # available: forward node -> its current representative (save or recompute).
    # As we process cheap ops in forward order, we update this mapping so
    # that later recomputations can reference earlier recomputed values.
    available = {fwd_node: sn for fwd_node, sn in save_nodes.items()}

    first_bwd = None
    for n in graph.nodes:
        if n.meta.get("phase") == "backward":
            first_bwd = n
            break

    if first_bwd is None:
        return PassResult(gm=gm)

    graph.inserting_before(first_bwd)

    for fwd_name in ordered:
        save_node, fwd_node = save_by_name[fwd_name]

        recompute_args = []
        for arg in fwd_node.args:
            if isinstance(arg, torch.fx.Node) and arg in available:
                recompute_args.append(available[arg])
            else:
                recompute_args.append(arg)

        recompute_node = graph.call_function(
            fwd_node.target, tuple(recompute_args)
        )
        recompute_node.name = graph._graph_namespace.create_name(
            f"recompute_{fwd_name}", None
        )
        _set_phase(recompute_node, "recompute")
        if "shape" in fwd_node.meta:
            recompute_node.meta["shape"] = fwd_node.meta["shape"]

        save_node.replace_all_uses_with(recompute_node)
        graph.erase_node(save_node)

        available[fwd_node] = recompute_node
        del save_by_name[fwd_name]

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def memory_plan(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    graph = gm.graph

    node_order = {}
    for i, node in enumerate(graph.nodes):
        node_order[node] = i

    # Compute liveness: for each tensor-producing call_function node,
    # record (creation_time, last_use_time, size_in_bytes).
    # Parameters and inputs are pre-allocated and excluded.
    intervals: Dict[torch.fx.Node, tuple] = {}
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        shape = node.meta.get("shape")
        if shape is None:
            continue

        creation = node_order[node]
        last_use = creation
        for user in node.users:
            t = node_order.get(user, creation)
            if t > last_use:
                last_use = t

        dtype = node.meta.get("dtype", torch.float32)
        bytes_per_elem = torch._utils._element_size(dtype)
        size = bytes_per_elem
        for s in shape:
            size *= s
        intervals[node] = (creation, last_use, size)

    if not intervals:
        _graph_meta(gm.graph)["memory_stats"] = {
            "naive_alloc": 0,
            "planned_alloc": 0,
            "num_tensors": 0,
            "num_pools": 0,
        }
        return PassResult(gm=gm)

    # Greedy best-fit pool assignment: reuse the smallest free pool that
    # can hold the new tensor. pools[i] = [capacity, free_after_time].
    pools: List[List[int]] = []
    assignments: Dict[torch.fx.Node, int] = {}

    for node in sorted(intervals, key=lambda n: intervals[n][0]):
        creation, last_use, size = intervals[node]

        best = None
        best_cap = float("inf")
        for i, (cap, free_after) in enumerate(pools):
            if free_after <= creation and cap >= size and cap < best_cap:
                best = i
                best_cap = cap

        if best is not None:
            pools[best][1] = last_use
            assignments[node] = best
        else:
            assignments[node] = len(pools)
            pools.append([size, last_use])

    naive_alloc = sum(sz for _, _, sz in intervals.values())
    planned_alloc = sum(cap for cap, _ in pools)

    for node, pool_id in assignments.items():
        node.meta["memory_pool"] = pool_id

    _graph_meta(gm.graph)["memory_stats"] = {
        "naive_alloc": naive_alloc,
        "planned_alloc": planned_alloc,
        "num_tensors": len(intervals),
        "num_pools": len(pools),
    }

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


# Ops eligible for Triton kernel fusion. See also _POINTWISE_PLACEMENT_OPS
# in collectives.py which lists ops whose DTensor placement propagates
# transparently — the two sets overlap but serve different purposes.
_POINTWISE_OPS = frozenset({
    "sin", "cos", "add", "sub", "mul", "div", "neg", "abs",
    "exp", "log", "relu", "sigmoid", "tanh", "rsqrt", "sqrt",
    "reciprocal", "where",
})


def _aten_op_name(target):
    packet = getattr(target, "overloadpacket", None)
    if packet is not None:
        return getattr(packet, "__name__", str(target))
    return getattr(target, "__name__", str(target))


def _node_shape(node):
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape"):
        return list(val.shape)
    return node.meta.get("shape")


def fuse(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    graph = gm.graph
    groups: List[FusionGroup] = []
    node_to_group: Dict[torch.fx.Node, FusionGroup] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        op_name = _aten_op_name(node.target)
        if op_name not in _POINTWISE_OPS:
            continue
        if node.kwargs:
            continue

        shape = _node_shape(node)

        merged = False
        for arg in node.args:
            if not isinstance(arg, torch.fx.Node) or arg not in node_to_group:
                continue
            # Don't merge through a node that has users outside the pointwise
            # universe — those users need the intermediate value to survive,
            # so the node can't be absorbed into a fused kernel.
            has_blocking_user = False
            for user in arg.users:
                if user is node:
                    continue
                if user.op != "call_function":
                    has_blocking_user = True
                    break
                if _aten_op_name(user.target) not in _POINTWISE_OPS:
                    has_blocking_user = True
                    break
            if has_blocking_user:
                continue
            group = node_to_group[arg]
            if group.shape is not None and group.shape == shape:
                group.node_names.append(node.name)
                group.op_names.append(op_name)
                group.output = node.name
                node_to_group[node] = group
                merged = True
                break

        if not merged:
            group = FusionGroup(
                group_id=len(groups),
                node_names=[node.name],
                op_names=[op_name],
                shape=shape,
                inputs=[],
                output=node.name,
            )
            groups.append(group)
            node_to_group[node] = group

    for group in groups:
        group_set = set(group.node_names)
        inputs = set()
        for n in graph.nodes:
            if n.name not in group_set:
                continue
            for arg in n.args:
                if isinstance(arg, torch.fx.Node) and arg.name not in group_set:
                    inputs.add(arg.name)
        group.inputs = sorted(inputs)

    multi_op = [g for g in groups if len(g.node_names) >= 2]

    name_to_node = {n.name: n for n in graph.nodes}
    counter = 0
    replaced = {}

    for group in multi_op:
        group.inputs = [replaced.get(inp, inp) for inp in group.inputs]
        group_set = set(group.node_names)

        input_index = {inp: i for i, inp in enumerate(group.inputs)}
        tmp_index: Dict[str, int] = {}

        fused_ops = []
        for idx, (node_name, op_name) in enumerate(
            zip(group.node_names, group.op_names)
        ):
            node = name_to_node[node_name]
            args = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name in input_index:
                        args.append(("input", input_index[arg.name]))
                    elif arg.name in tmp_index:
                        args.append(("tmp", tmp_index[arg.name]))
                else:
                    args.append(("const", arg))
            fused_ops.append(FusedOp(op_name=op_name, args=args))
            tmp_index[node_name] = idx

        kernel_name = "fused_" + "_".join(group.op_names[:4])
        if len(group.op_names) > 4:
            kernel_name += f"_x{len(group.op_names)}"
        kernel_name = f"{kernel_name}_{counter}"
        counter += 1

        kernel = FusedKernel(
            name=kernel_name,
            ops=fused_ops,
            n_inputs=len(group.inputs),
            shape=group.shape,
        )

        input_nodes = [name_to_node[inp] for inp in group.inputs]
        output_node = name_to_node[group.output]

        graph.inserting_before(output_node)
        fused_node = graph.call_function(kernel, tuple(input_nodes))
        fused_node.name = graph._graph_namespace.create_name(kernel_name, None)
        _set_phase(fused_node, output_node.meta.get("phase", "forward"))
        if group.shape:
            fused_node.meta["shape"] = group.shape

        output_node.replace_all_uses_with(fused_node)
        replaced[group.output] = fused_node.name
        name_to_node[fused_node.name] = fused_node

        for node_name in reversed(group.node_names):
            node = name_to_node[node_name]
            if not node.users:
                del name_to_node[node_name]
                graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def subclass_unwrap(gm, example_inputs, *, world_size=2):
    graph = gm.graph

    placement = {}
    for node in graph.nodes:
        if node.op not in ("get_attr", "placeholder"):
            continue
        spec = node.meta.get("dtensor_spec")
        if spec:
            placement[node] = spec

    if not placement:
        return PassResult(gm=gm)

    for node in graph.nodes:
        if node in placement or node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")

        if name in _POINTWISE_PLACEMENT_OPS:
            _propagate_pointwise(node, placement)
        elif name in _BINARY_ELEMENTWISE_OPS:
            _propagate_binary(node, placement)
        elif name in _MATMUL_OPS:
            _propagate_matmul(node, placement)
        elif name in _REDUCTION_OPS:
            _propagate_reduction(node, placement)
        elif name in _VIEW_OPS:
            _propagate_view(node, placement)
        elif name in _NORM_OPS:
            _propagate_norm(node, placement)
        elif name == "embedding":
            _propagate_embedding(node, placement)

    # Nodes that have been created by collective insertion and should not
    # have their shapes adjusted in the final shard-dim division pass.
    collective_nodes = set()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        spec = placement.get(node)
        if spec is None or spec[0] != "_Partial":
            continue
        users = list(node.users)
        graph.inserting_after(node)
        ar = graph.call_function(_allreduce, (node,))
        ar.name = graph._graph_namespace.create_name("allreduce", None)
        _set_phase(ar, node.meta.get("phase", "forward"))
        if "shape" in node.meta:
            ar.meta["shape"] = list(node.meta["shape"])
        for user in users:
            if user.op == "call_function" and user.target is torch.Tensor.copy_:
                continue
            user.replace_input_with(node, ar)
        placement[ar] = ("Replicate", None)
        collective_nodes.add(ar)

    _BINARY_ALLGATHER_OPS = frozenset({
        "sub", "add", "mul", "div", "pow",
        "where", "maximum", "minimum",
        "eq", "ne", "lt", "le", "gt", "ge",
    })

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if name not in _BINARY_ALLGATHER_OPS:
            continue

        tensor_args = [
            (a, placement.get(a))
            for a in node.args
            if isinstance(a, torch.fx.Node)
        ]
        if len(tensor_args) < 2:
            continue

        a_node, a_spec = tensor_args[0]
        b_node, b_spec = tensor_args[1]

        target_node = None
        target_spec = None
        if _is_shard(a_spec) and (_is_replicate(b_spec) or b_spec is None):
            target_node = a_node
            target_spec = a_spec
        elif _is_shard(b_spec) and (_is_replicate(a_spec) or a_spec is None):
            target_node = b_node
            target_spec = b_spec

        if target_node is not None:
            graph.inserting_before(node)
            ag = graph.call_function(_allgather, (target_node,))
            ag.name = graph._graph_namespace.create_name("allgather", None)
            _set_phase(ag, node.meta.get("phase", "forward"))
            # Allgather output has the full (global) shape: restore the
            # shard dim to its pre-sharding size.
            if "shape" in target_node.meta:
                gathered_shape = list(target_node.meta["shape"])
                shard_dim = target_spec[1]
                gathered_shape[shard_dim] = gathered_shape[shard_dim] * world_size
                ag.meta["shape"] = gathered_shape
            node.replace_input_with(target_node, ag)
            placement[ag] = ("Replicate", None)
            placement[node] = ("Replicate", None)
            collective_nodes.add(ag)

    for node in graph.nodes:
        spec = placement.get(node)
        shape = node.meta.get("shape")
        if spec:
            node.meta["dtensor_spec"] = spec
        if node in collective_nodes:
            continue
        if spec and shape and _is_shard(spec):
            dim = spec[1]
            local_shape = list(shape)
            if local_shape[dim] % world_size != 0:
                raise ValueError(
                    f"subclass_unwrap: node '{node.name}' shard dim {dim} "
                    f"(size {local_shape[dim]}) is not evenly divisible by "
                    f"world_size={world_size}."
                )
            local_shape[dim] = local_shape[dim] // world_size
            node.meta["shape"] = local_shape

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


# ── Per-op backward ──────────────────────────────────────────────────────────


def autograd_per_op(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Reverse-mode AD via dispatcher-based tracing through torch.autograd.grad.

    Executes the forward graph on real tensors with requires_grad=True for
    params, then traces backward under _BackwardRecorder to capture all ATen
    ops emitted by autograd. This automatically handles every op that PyTorch's
    autograd supports, with no per-op whitelist.
    """
    graph = gm.graph

    param_targets = []
    param_nodes = {}
    for n in graph.nodes:
        if n.op == "get_attr":
            param_targets.append(n.target)
            param_nodes[n.target] = n

    if not param_targets:
        return PassResult(gm=gm)

    output_node = None
    for n in graph.nodes:
        if n.op == "output":
            output_node = n
            break

    orig_output = output_node.args[0]
    has_multi = isinstance(orig_output, (tuple, list))
    loss_node = orig_output[0] if has_multi else orig_output

    # Execute forward graph node-by-node on real tensors to build the
    # autograd tape. Params get requires_grad=True; inputs are detached.
    # Forward execution uses _ForwardDecomposer for compound ops (those
    # whose backward references internal intermediates like layer_norm's
    # mean/rstd) — this decomposes them into ATen ops so all saved tensors
    # are explicit FX nodes.
    node_to_tensor = {}
    id_to_node = {}
    storage_to_node = {}
    dptr_to_node = {}
    _live_tensors = []

    def _track(tensor, node):
        id_to_node[id(tensor)] = node
        storage_to_node[_storage_key(tensor)] = node
        if tensor.data_ptr() not in dptr_to_node:
            dptr_to_node[tensor.data_ptr()] = node
        _live_tensors.append(tensor)

    # Snapshot nodes before iteration: _ForwardDecomposer inserts new
    # ATen nodes into the graph, and we must not iterate over those.
    placeholder_nodes = [
        n for n in graph.nodes if n.op == "placeholder"
    ]
    ph_to_idx = {n: i for i, n in enumerate(placeholder_nodes)}
    orig_nodes = list(graph.nodes)

    for node in orig_nodes:
        if node.op == "placeholder":
            t = example_inputs[ph_to_idx[node]].detach().clone()
            node_to_tensor[node] = t
            _track(t, node)

        elif node.op == "get_attr":
            param = _deep_getattr(gm, node.target)
            t = param.detach().clone().requires_grad_(True)
            node_to_tensor[node] = t
            _track(t, node)

        elif node.op == "call_function":
            phase = node.meta.get("phase", "forward")
            if phase == "copy-back":
                continue

            def _resolve(x):
                if isinstance(x, torch.fx.Node):
                    return node_to_tensor.get(x, x)
                if isinstance(x, (list, tuple)):
                    return type(x)(_resolve(i) for i in x)
                return x

            args = tuple(_resolve(a) for a in node.args)
            kwargs = {k: _resolve(v) for k, v in node.kwargs.items()}

            # Execute under _ForwardDecomposer to intercept internal ATen
            # ops from compound forward ops (rms_norm, layer_norm, etc.).
            # This records their intermediates as FX nodes so backward
            # can reference them. The high-level FX node is kept as-is;
            # the decomposed nodes are "extra" forward nodes.
            # Use inserting_before(output_node) so nodes are in correct
            # topological order (inserting_after would reverse them).
            decomp_nodes = []
            graph.inserting_before(output_node)
            decomposer = _ForwardDecomposer(
                graph, id_to_node, storage_to_node,
                decomp_nodes, _live_tensors, node.meta,
            )

            with decomposer:
                result = node.target(*args, **kwargs)

            # Update dptr_to_node for tensors tracked by the decomposer
            for sk, n in storage_to_node.items():
                dptr = sk[0]
                if dptr not in dptr_to_node:
                    dptr_to_node[dptr] = n

            if isinstance(result, torch.Tensor):
                node_to_tensor[node] = result
                _track(result, node)
            elif isinstance(result, (tuple, list)):
                node_to_tensor[node] = result
                for r in result:
                    if isinstance(r, torch.Tensor):
                        _live_tensors.append(r)

        elif node.op == "output":
            continue

    loss_tensor = node_to_tensor[loss_node]
    param_tensors = [node_to_tensor[param_nodes[t]] for t in param_targets]

    graph.inserting_before(output_node)

    with _BackwardRecorder(graph, id_to_node, storage_to_node,
                           dptr_to_node, _live_tensors):
        grads = torch.autograd.grad(
            loss_tensor, param_tensors, allow_unused=True,
        )

    param_grad_info = {}
    grad_nodes = []
    for i, (target, g) in enumerate(zip(param_targets, grads)):
        if g is not None:
            grad_node = id_to_node.get(id(g))
        else:
            grad_node = None
        if grad_node is None:
            pn = param_nodes[target]
            grad_node = graph.call_function(torch.zeros_like, (pn,))
            grad_node.name = graph._graph_namespace.create_name(
                f"grad_{target.split('.')[-1]}", None
            )
            _set_phase(grad_node, "backward")
            shape = pn.meta.get("shape")
            if shape is not None:
                grad_node.meta["shape"] = list(shape)
        param_grad_info[target] = i
        grad_nodes.append(grad_node)

    output_node.args = (tuple([loss_node] + grad_nodes),)
    _graph_meta(gm.graph)["param_grad_info"] = param_grad_info

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


# ── Dynamize ─────────────────────────────────────────────────────────────────


def _align_reshape(in_shape, out_shape):
    """Map input dims to output dims for a reshape operation.

    Returns {input_dim: [output_dims...]}. Walks both shapes left-to-right,
    matching sizes 1:1 when equal, splitting an input dim across multiple
    output dims when the input dim is larger, and merging multiple input
    dims into one output dim when the output dim is larger.
    """
    result = {}
    i, o = 0, 0
    while i < len(in_shape) and o < len(out_shape):
        if in_shape[i] == out_shape[o]:
            result[i] = [o]
            i += 1
            o += 1
        elif in_shape[i] > out_shape[o]:
            start = o
            prod = out_shape[o]
            o += 1
            while prod < in_shape[i] and o < len(out_shape):
                prod *= out_shape[o]
                o += 1
            result[i] = list(range(start, o))
            i += 1
        else:
            start = i
            prod = in_shape[i]
            i += 1
            while prod < out_shape[o] and i < len(in_shape):
                prod *= in_shape[i]
                i += 1
            for idx in range(start, i):
                result[idx] = [o]
            o += 1
    return result


def dynamize(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
) -> PassResult:
    """Make specified dimensions dynamic by inserting explicit size-extraction
    nodes and replacing concrete shape literals in reshape/view ops."""
    graph = gm.graph

    placeholders = {}
    for n in graph.nodes:
        if n.op == "placeholder":
            placeholders[n.name] = n

    if dynamic_dims is None:
        dynamic_dims = {name: [0] for name in placeholders}

    first_call = next((n for n in graph.nodes if n.op == "call_function"), None)
    if first_call is None:
        return PassResult(gm=gm)

    graph.inserting_before(first_call)

    size_nodes = {}
    batch_size_node = None
    batch_concrete = None
    for ph_name, dims in dynamic_dims.items():
        ph = placeholders.get(ph_name)
        if ph is None:
            continue
        shape = ph.meta.get("shape", [])
        for d in dims:
            if d >= len(shape):
                continue
            sn = graph.call_function(torch.Tensor.size, (ph, d))
            sn.name = graph._graph_namespace.create_name(f"size_{ph_name}_d{d}", None)
            _set_phase(sn, "forward")
            size_nodes[(ph_name, d)] = sn
            if d == 0 and batch_size_node is None:
                batch_size_node = sn
                batch_concrete = shape[d]

    if batch_size_node is None:
        graph.lint()
        gm.recompile()
        return PassResult(gm=gm)

    dyn: Dict[torch.fx.Node, set] = {}
    for ph_name, dims in dynamic_dims.items():
        ph = placeholders.get(ph_name)
        if ph:
            dyn[ph] = set(dims)

    _INHERIT_OPS = frozenset({
        "sin", "cos", "add", "sub", "mul", "div", "neg", "abs",
        "pow", "dropout", "softmax", "rms_norm", "clone",
        "contiguous", "matmul", "mm",
    })

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        op = getattr(node.target, "__name__", "")

        primary = None
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg in dyn:
                primary = arg
                break
        if primary is None:
            continue
        pd = dyn[primary]

        if op in _INHERIT_OPS:
            dyn[node] = set(pd)

        elif op == "transpose":
            d0 = node.args[1] if len(node.args) > 1 else 0
            d1 = node.args[2] if len(node.args) > 2 else 1
            new_d = set()
            for d in pd:
                if d == d0:
                    new_d.add(d1)
                elif d == d1:
                    new_d.add(d0)
                else:
                    new_d.add(d)
            dyn[node] = new_d

        elif op in ("reshape", "view"):
            input_shape = primary.meta.get("shape")
            output_shape = node.meta.get("shape")
            if input_shape and output_shape:
                mapping = _align_reshape(input_shape, output_shape)
                new_d = set()
                for in_dim, out_dims in mapping.items():
                    if in_dim in pd:
                        new_d.update(out_dims)
                dyn[node] = new_d

                # Tensor.reshape takes varargs: args = (tensor, d0, d1, ...)
                # torch.reshape takes a tuple: args = (tensor, (d0, d1, ...))
                if len(node.args) > 2:
                    shape_dims = list(node.args[1:])
                else:
                    shape_dims = (
                        list(node.args[1])
                        if isinstance(node.args[1], (list, tuple))
                        else None
                    )
                if shape_dims is not None:
                    changed = False
                    for in_dim, out_dims in mapping.items():
                        if in_dim not in pd or len(out_dims) != 1:
                            continue
                        od = out_dims[0]
                        if (
                            isinstance(shape_dims[od], int)
                            and shape_dims[od] == batch_concrete
                        ):
                            shape_dims[od] = batch_size_node
                            changed = True
                    if changed:
                        if len(node.args) > 2:
                            node.args = (node.args[0], *shape_dims)
                        else:
                            node.args = (node.args[0], tuple(shape_dims))

        elif op == "mean":
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            if dim is None:
                pass
            else:
                dyn[node] = {d for d in pd if d != dim}

        node.meta["dynamic_dims"] = dyn.get(node, set())

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def fsdp_unwrap(gm, example_inputs, *, dp_dim=0, world_size=2):
    """FSDP pass: insert allgather before forward uses of DP-sharded parameters
    and reduce_scatter after backward gradient computation.

    Parameters sharded on `dp_dim` (Shard(dp_dim)) are identified as FSDP
    parameters. The pass:
    1. Inserts allgather before forward parameter usage
    2. Inserts reduce_scatter after backward gradient computation
    3. Handles reshard_after_forward: frees full params after forward
    """
    graph = gm.graph

    fsdp_params = {}
    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        spec = node.meta.get("dtensor_spec")
        if spec is not None and spec[0] == "Shard" and spec[1] == dp_dim:
            fsdp_params[node] = spec

    if not fsdp_params:
        return PassResult(gm=gm)

    for param_node in list(fsdp_params):
        fwd_users = [
            u for u in param_node.users
            if u.op == "call_function"
            and u.meta.get("phase", "forward") == "forward"
        ]

        if not fwd_users:
            continue

        first_user = fwd_users[0]
        graph.inserting_before(first_user)
        ag = graph.call_function(_allgather, (param_node,))
        ag.name = graph._graph_namespace.create_name(
            f"fsdp_ag_{param_node.target.split('.')[-1]}", None
        )
        _set_phase(ag, "forward")
        if "shape" in param_node.meta:
            shape = list(param_node.meta["shape"])
            shape[dp_dim] = shape[dp_dim] * world_size
            ag.meta["shape"] = shape
        ag.meta["dtensor_spec"] = ("Replicate", None)

        for user in fwd_users:
            user.replace_input_with(param_node, ag)

    param_grad_info = _graph_meta(gm.graph).get("param_grad_info", {})
    if param_grad_info:
        output_node = None
        for n in graph.nodes:
            if n.op == "output":
                output_node = n
                break

        orig_output = output_node.args[0]
        if isinstance(orig_output, (tuple, list)):
            param_targets_set = {n.target for n in fsdp_params}
            output_replacements = {}
            for param_name, grad_idx in param_grad_info.items():
                if param_name not in param_targets_set:
                    continue
                grad_node = orig_output[1 + grad_idx]
                if not isinstance(grad_node, torch.fx.Node):
                    continue

                users = list(grad_node.users)
                graph.inserting_after(grad_node)
                rs = graph.call_function(_reduce_scatter, (grad_node,))
                rs.name = graph._graph_namespace.create_name(
                    f"fsdp_rs_{param_name.split('.')[-1]}", None
                )
                _set_phase(rs, "backward")
                if "shape" in grad_node.meta:
                    shape = list(grad_node.meta["shape"])
                    if shape[dp_dim] % world_size != 0:
                        raise ValueError(
                            f"fsdp_unwrap: grad for '{param_name}' dim {dp_dim} "
                            f"(size {shape[dp_dim]}) is not evenly divisible by "
                            f"world_size={world_size}."
                        )
                    shape[dp_dim] = shape[dp_dim] // world_size
                    rs.meta["shape"] = shape
                rs.meta["dtensor_spec"] = ("Shard", dp_dim)

                for user in users:
                    if user is not rs and user.op != "output":
                        user.replace_input_with(grad_node, rs)

                output_replacements[1 + grad_idx] = rs

            if output_replacements:
                new_output = list(orig_output)
                for idx, rs in output_replacements.items():
                    new_output[idx] = rs
                output_node.args = (tuple(new_output),)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


# ── Triton code generation ────────────────────────────────────────────────────


_TRITON_OP_MAP = {
    "sin": ("tl.math.sin({})", 1),
    "cos": ("tl.math.cos({})", 1),
    "exp": ("tl.math.exp({})", 1),
    "log": ("tl.math.log({})", 1),
    "abs": ("tl.abs({})", 1),
    "neg": ("-({})", 1),
    "sqrt": ("tl.math.sqrt({})", 1),
    "rsqrt": ("tl.math.rsqrt({})", 1),
    "sigmoid": ("tl.sigmoid({})", 1),
    "tanh": ("tl.math.tanh({})", 1),
    "relu": ("tl.maximum({}, 0.0)", 1),
    "add": ("({} + {})", 2),
    "sub": ("({} - {})", 2),
    "mul": ("({} * {})", 2),
    "div": ("({} / {})", 2),
    "where": ("tl.where({}, {}, {})", 3),
    "reciprocal": ("(1.0 / {})", 1),
}


def triton_codegen(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Generate Triton GPU kernel source code for fused ops in the graph.

    Walks the graph looking for FusedKernel nodes (created by the fuse pass)
    and emits Triton JIT kernel code for each one. The generated code is
    stored in the graph's metadata under the key "triton_code".
    """
    kernels = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, FusedKernel):
            continue

        kernel = node.target
        in_ptrs = [f"in_ptr{i}" for i in range(kernel.n_inputs)]
        params = ", ".join(
            in_ptrs + ["out_ptr", "n_elements", "BLOCK_SIZE: tl.constexpr = 1024"]
        )

        lines = [
            "@triton.jit",
            f"def {kernel.name}({params}):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_elements",
            "",
        ]

        for i in range(kernel.n_inputs):
            lines.append(f"    x{i} = tl.load(in_ptr{i} + offs, mask=mask)")
        lines.append("")

        val_map: Dict[tuple, str] = {}
        for i in range(kernel.n_inputs):
            val_map[("input", i)] = f"x{i}"

        for tmp_idx, op in enumerate(kernel.ops):
            entry = _TRITON_OP_MAP.get(op.op_name)
            if entry is None:
                continue
            template, nargs = entry

            arg_vars = []
            for arg in op.args:
                key = (arg[0], arg[1])
                if key in val_map:
                    arg_vars.append(val_map[key])
                elif arg[0] == "const":
                    arg_vars.append(str(arg[1]))
                else:
                    arg_vars.append("???")

            result = f"tmp{tmp_idx}"

            if nargs == 1 and arg_vars:
                expr = template.format(arg_vars[0])
            elif nargs == 2 and len(arg_vars) >= 2:
                expr = template.format(arg_vars[0], arg_vars[1])
            elif nargs == 3 and len(arg_vars) >= 3:
                expr = template.format(arg_vars[0], arg_vars[1], arg_vars[2])
            else:
                expr = f"# {op.op_name}({', '.join(arg_vars)})"

            lines.append(f"    {result} = {expr}")
            val_map[("tmp", tmp_idx)] = result

        lines.append("")
        if kernel.ops:
            lines.append(
                f"    tl.store(out_ptr + offs, tmp{len(kernel.ops) - 1}, mask=mask)"
            )
        else:
            lines.append("    pass")

        kernels.append("\n".join(lines))

    if not kernels:
        code = "# No fused kernels found\n"
    else:
        code = "\n\n\n".join(kernels) + "\n"

    _graph_meta(gm.graph)["triton_code"] = code
    return PassResult(gm=gm)


# Ops that perform CPU-side work or autograd calls and cannot be captured
# inside a CUDA graph. After fixing adamw_step to avoid .item(), it is
# capturable, so it is NOT listed here.
_CUDAGRAPH_NON_CAPTURABLE = frozenset({
    "save_rng_state",
    "load_rng_state",
    "_rms_norm_bwd",
})


def cudagraph_partition(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Analyze the graph for CUDA-graph compatibility and segment it.

    Walks all call_function nodes and classifies each as capturable or not.
    Non-capturable nodes (CPU RNG state ops, autograd-based backward ops)
    act as segment boundaries. Contiguous runs of capturable nodes are
    assigned a segment_id stored in node.meta["cudagraph_segment"].

    Errors if any node has non-empty dynamic_dims metadata, since CUDA
    graphs require static shapes.
    """
    for node in gm.graph.nodes:
        if node.meta.get("dynamic_dims"):
            raise RuntimeError(
                f"cudagraph_partition: node '{node.name}' has dynamic_dims "
                f"{node.meta['dynamic_dims']}. CUDA graphs require static shapes."
            )

    segment_id = 0
    in_segment = False
    segments = {}
    segment_start = None
    segment_count = 0
    prev_node_name = None

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        name = getattr(node.target, "__name__", "")
        capturable = name not in _CUDAGRAPH_NON_CAPTURABLE

        if capturable:
            if not in_segment:
                in_segment = True
                segment_start = node.name
                segment_count = 0
            node.meta["cudagraph_segment"] = segment_id
            segment_count += 1
        else:
            if in_segment:
                segments[segment_id] = {
                    "start": segment_start,
                    "end": prev_node_name,
                    "num_nodes": segment_count,
                }
                segment_id += 1
                in_segment = False

        prev_node_name = node.name

    if in_segment:
        segments[segment_id] = {
            "start": segment_start,
            "end": prev_node_name,
            "num_nodes": segment_count,
        }

    _graph_meta(gm.graph)["cudagraph_segments"] = segments
    return PassResult(gm=gm)


def precompile(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Generate a standalone Python module from the compiled graph.

    Emits a self-contained Python file with Triton kernels (if any) and a
    CompiledModule class that reproduces the graph's computation. The
    generated code is stored in graph metadata under "precompiled_code".
    """
    triton_code = _graph_meta(gm.graph).get("triton_code", "")
    lines = ["import torch", ""]

    has_torchlite_ops = any(
        n.op == "call_function"
        and "torch._torchlite.ops" in getattr(n.target, "__module__", "")
        for n in gm.graph.nodes
    )
    has_torchlite_collectives = any(
        n.op == "call_function"
        and "torch._torchlite.collectives" in getattr(n.target, "__module__", "")
        for n in gm.graph.nodes
    )
    if has_torchlite_ops:
        lines.append("from torch._torchlite import ops as torchlite_ops")
    if has_torchlite_collectives:
        lines.append("from torch._torchlite import collectives as torchlite_collectives")
    if has_torchlite_ops or has_torchlite_collectives:
        lines.append("")

    has_triton = triton_code.strip() and triton_code.strip() != "# No fused kernels found"
    if has_triton:
        lines += ["import triton", "import triton.language as tl", "", ""]
        lines.append(triton_code.rstrip())
        lines += ["", ""]

    lines.append("class CompiledModule:")
    lines.append("    def __init__(self, state_dict):")
    lines.append("        self.state_dict = state_dict")
    lines.append("")
    lines.append("")
    lines.append("    def __call__(self, *args, **kwargs):")
    lines.append("        return self.forward(*args, **kwargs)")
    lines.append("")

    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node.name)

    sig = ", ".join(["self"] + placeholders)
    lines.append(f"    def forward({sig}):")

    def _fmt(a):
        if isinstance(a, torch.fx.Node):
            return a.name
        return repr(a)

    def _fmt_container(v):
        if isinstance(v, torch.fx.Node):
            return v.name
        if isinstance(v, tuple):
            return "(" + ", ".join(_fmt_container(x) for x in v) + ("," if len(v) == 1 else "") + ")"
        if isinstance(v, list):
            return "[" + ", ".join(_fmt_container(x) for x in v) + "]"
        if isinstance(v, dict):
            items = ", ".join(f"{repr(k)}: {_fmt_container(x)}" for k, x in v.items())
            return "{" + items + "}"
        return repr(v)

    def _fmt_args(node):
        parts = [_fmt_container(a) for a in node.args]
        for k, v in (node.kwargs or {}).items():
            parts.append(f"{k}={_fmt_container(v)}")
        return ", ".join(parts)

    def _emit_target_expr(target):
        if isinstance(target, torch._ops.OpOverload):
            schema = target._schema.name  # e.g. aten::add
            ns, op = schema.split("::", 1)
            overload = target._overloadname or "default"
            return f"torch.ops.{ns}.{op}.{overload}"

        resolved = resolve_name(target)
        if resolved is not None and resolved.startswith("torch.Tensor."):
            method = resolved.split(".")[-1]
            return f"{method}__METHOD__"

        module = getattr(target, "__module__", "")
        fn_name = getattr(target, "__name__", str(target))
        if "torch._torchlite.ops" in module:
            return f"torchlite_ops.{fn_name}"
        if "torch._torchlite.collectives" in module:
            return f"torchlite_collectives.{fn_name}"
        return f"torch.{fn_name}"

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        elif node.op == "get_attr":
            lines.append(f"        {node.name} = self.state_dict['{node.target}']")
        elif node.op == "call_function":
            target = node.target
            if isinstance(target, FusedKernel):
                input_nodes = [a for a in node.args if isinstance(a, torch.fx.Node)]
                in_args = ", ".join(a.name for a in input_nodes)
                shape = target.shape
                numel = 1
                for s in (shape or []):
                    numel *= s
                device_ref = input_nodes[0].name if input_nodes else None
                dtype = node.meta.get("dtype", torch.float32)
                dtype_str = str(dtype)
                if device_ref:
                    lines.append(
                        f"        {node.name} = torch.empty({shape}, dtype={dtype_str}, device={device_ref}.device)"
                    )
                else:
                    lines.append(f"        {node.name} = torch.empty({shape}, dtype={dtype_str})")
                lines.append(
                    f"        {target.name}[(({numel} + 1023) // 1024,)]"
                    f"({in_args}, {node.name}, {numel})"
                )
            else:
                if target is operator.getitem:
                    lines.append(f"        {node.name} = {_fmt(node.args[0])}[{_fmt(node.args[1])}]")
                    continue
                elif target is torch.Tensor.copy_:
                    lines.append(f"        {_fmt(node.args[0])}.copy_({_fmt(node.args[1])})")
                    continue
                expr = _emit_target_expr(target)
                if expr.endswith("__METHOD__"):
                    method = expr.replace("__METHOD__", "")
                    obj = _fmt_container(node.args[0])
                    arg_parts = [_fmt_container(a) for a in node.args[1:]]
                    arg_parts += [
                        f"{k}={_fmt_container(v)}"
                        for k, v in (node.kwargs or {}).items()
                    ]
                    args_str = ", ".join(arg_parts)
                    lines.append(f"        {node.name} = {obj}.{method}({args_str})")
                else:
                    args_str = _fmt_args(node)
                    lines.append(f"        {node.name} = {expr}({args_str})")
        elif node.op == "output":
            args = node.args[0]
            if isinstance(args, (tuple, list)):
                ret = ", ".join(_fmt(a) for a in args)
                lines.append(f"        return ({ret})")
            else:
                lines.append(f"        return {_fmt(args)}")
    lines.append("")

    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    import sys")
    lines.append("    state_dict = torch.load(sys.argv[1]) if len(sys.argv) > 1 else {}")
    lines.append("    mod = CompiledModule(state_dict)")
    ph_shapes = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ph_shapes[node.name] = node.meta.get("shape", [1])
    in_str = ", ".join(f"torch.randn({ph_shapes.get(p, [1])})" for p in placeholders)
    lines.append(f"    result = mod.forward({in_str})")
    lines.append("    print('Result:', result)")

    code = "\n".join(lines) + "\n"
    _graph_meta(gm.graph)["precompiled_code"] = code
    return PassResult(gm=gm)
