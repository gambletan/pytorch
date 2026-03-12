"""Dynamize pass: make batch dimensions dynamic."""
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _graph_meta,
    _set_phase,
    _VARARGS_TENSOR_METHODS,
    PassResult,
)


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
            sn.name = _create_name(graph, f"size_{ph_name}_d{d}")
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
        "relu", "sigmoid", "tanh", "exp", "log",
        "rsqrt", "sqrt", "reciprocal",
        "gelu", "silu", "layer_norm",
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
                        if isinstance(shape_dims[od], int):
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

