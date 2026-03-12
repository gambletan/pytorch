"""Decomposition pass: lower compound ops into core ATen ops."""
import logging
import operator
from typing import Dict, List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _is_torch_op,
    _iter_node_args,
    _PROVENANCE_KEYS,
    _set_phase,
    FusedKernel,
    PassResult,
)

log = logging.getLogger(__name__)


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

        if func is torch.ops.aten.view.default:
            func = torch.ops.aten.reshape.default

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
        if func is torch.ops.aten._to_copy.default:
            dev = fx_kwargs.get("device")
            if isinstance(dev, torch.device) and dev.type == "meta":
                del fx_kwargs["device"]

        new_node = self.graph.call_function(func, fx_args, fx_kwargs)
        pkt = getattr(func, "overloadpacket", None)
        name_hint = getattr(pkt, "__name__", None) if pkt else None
        if name_hint is None:
            name_hint = getattr(func, "__name__", "decomp")
        new_node.name = _create_name(self.graph, name_hint)

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
                    get_node.name = _create_name(self.graph, f"{name_hint}_item")
                    get_node.meta["shape"] = list(r.shape)
                    get_node.meta["dtype"] = r.dtype
                    self.id_to_node[id(r)] = get_node
                    self.new_nodes.append(get_node)

        for k, v in self.prov.items():
            new_node.meta[k] = v

        self.new_nodes.append(new_node)
        return result


import torch.nn.functional as _F

_DECOMP_BLOCKLIST = frozenset({
    torch.ops.aten.reshape.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    _F.rms_norm,
    _F.scaled_dot_product_attention,
})


def decompose(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    from torch._decomp import core_aten_decompositions

    graph = gm.graph
    decomp_table = core_aten_decompositions()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target in _DECOMP_BLOCKLIST:
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
                decomp_result = node.target(*real_args, **real_kwargs)
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

        # For multi-output ops like native_layer_norm, the decomposition
        # returns a tuple of tensors.  The original graph has getitem nodes
        # that extract each element.  We need to replace each getitem with
        # the corresponding decomposed tensor's FX node, not wholesale-replace
        # the multi-output op with a single node (which would make the
        # getitem nodes index into the wrong value at runtime).
        if isinstance(decomp_result, (tuple, list)):
            result_map = {}
            for i, r in enumerate(decomp_result):
                if isinstance(r, torch.Tensor) and id(r) in id_to_node:
                    result_map[i] = id_to_node[id(r)]
            for user in list(node.users.keys()):
                if (
                    user.op == "call_function"
                    and user.target is operator.getitem
                    and len(user.args) >= 2
                ):
                    idx = user.args[1]
                    replacement = result_map.get(idx)
                    if replacement is not None:
                        user.replace_all_uses_with(replacement)
                        graph.erase_node(user)
            if not node.users:
                graph.erase_node(node)
        elif isinstance(decomp_result, torch.Tensor):
            result_node = id_to_node.get(id(decomp_result), new_nodes[-1])
            node.replace_all_uses_with(result_node)
            graph.erase_node(node)
        else:
            result_node = new_nodes[-1]
            node.replace_all_uses_with(result_node)
            graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
