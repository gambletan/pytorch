"""Activation checkpoint and save_activations passes."""
from typing import List, Optional

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _aten_op_name,
    _create_name,
    _iter_node_args,
    _set_phase,
    PassResult,
)
from torch._torchlite.ops import _save_for_backward


def save_activations(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Save forward activations consumed by backward nodes.

    Inserts explicit save_for_backward identity nodes at the
    forward/backward boundary for every forward activation that is
    read by at least one backward node. Parameters and inputs are
    excluded since they are always available.
    """
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
        save_node.name = _create_name(graph, f"save_{fwd_node.name}")
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
        recompute_node.name = _create_name(graph, f"recompute_{fwd_name}")
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
