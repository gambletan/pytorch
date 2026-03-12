"""Autograd per-op pass: reverse-mode AD via dispatcher tracing."""
import operator
import warnings
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _deep_getattr,
    _deep_setattr,
    _graph_meta,
    _is_torch_op,
    _iter_node_args,
    _PROVENANCE_KEYS,
    _set_phase,
    FusedKernel,
    PassResult,
)
from torch._torchlite.ops import _save_for_backward


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
            # Validate that the source node's storage can contain the
            # requested view. Two tensors with aliased storage but
            # incompatible layouts would produce a wrong as_strided.
            src_shape = src_node.meta.get("shape")
            if src_shape is not None:
                src_numel = 1
                for s in src_shape:
                    src_numel *= s
                view_max = x.storage_offset()
                for dim_size, stride in zip(x.shape, x.stride()):
                    if dim_size > 0:
                        view_max += (dim_size - 1) * stride
                if view_max >= src_numel:
                    return None
            view_node = self.graph.call_function(
                torch.ops.aten.as_strided.default,
                (src_node, list(x.shape), list(x.stride()), x.storage_offset()),
            )
            view_node.name = _create_name(self.graph, "bwd_view")
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
        new_node.name = _create_name(self.graph, name_hint)
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
                    get_node.name = _create_name(self.graph, f"{name_hint}_item")
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
        new_node.name = _create_name(self.graph, name_hint)
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
                    get_node.name = _create_name(self.graph, f"{name_hint}_item")
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    _set_phase(get_node, "forward")
                    self.id_to_node[id(r)] = get_node
                    self.storage_to_node[_storage_key(r)] = get_node
                    self.live_tensors.append(r)
                    self.decomp_nodes.append(get_node)

        self.decomp_nodes.append(new_node)
        return result


def autograd_per_op(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Reverse-mode AD via dispatcher-based tracing through torch.autograd.grad.

    Executes the forward graph on real tensors with requires_grad=True for
    params, then traces backward under _BackwardRecorder to capture all ATen
    ops emitted by autograd. This automatically handles every op that PyTorch's
    autograd supports, with no per-op whitelist.

    WARNING: This pass re-executes the full forward graph on real tensors to
    build the autograd tape, which allocates real GPU/CPU memory proportional
    to the model size and batch size. For large models this can be significant.
    A proper fix would use meta-tensors or symbolic autograd tracing, but that
    is out of scope for now.
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
            grad_node.name = _create_name(
                graph, f"grad_{target.split('.')[-1]}"
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

