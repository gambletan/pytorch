"""Graph normalization passes."""
import operator
from typing import List

import torch
from torch.fx import GraphModule
from torch.overrides import resolve_name

from torch._torchlite.passes.common import (
    _create_name,
    _deep_getattr,
    _DUNDER_INPLACE,
    _DUNDER_TO_OP,
    _REVERSE_DUNDERS,
    _VARARGS_TENSOR_METHODS,
    PassResult,
)


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
            node.name = _create_name(graph, name)
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
