"""RNG functionalization pass."""
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _set_phase,
    PassResult,
)
from torch._torchlite.ops import _save_rng_state, _load_rng_state


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
        state_node.name = _create_name(graph, "rng_state")
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
        restore.name = _create_name(graph, "restore_rng")
        _set_phase(restore, "backward")
        restore.meta["bwd_of"] = node.meta.get("bwd_of")

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
