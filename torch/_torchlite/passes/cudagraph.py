"""CUDA graph partitioning pass."""
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _graph_meta,
    _set_phase,
    FusedKernel,
    PassResult,
)
from torch._torchlite.ops import _save_rng_state, _load_rng_state


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
        if node.op not in ("call_function", "call_module"):
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
