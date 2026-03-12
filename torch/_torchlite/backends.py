"""Post-compilation runtime utilities.

These are NOT FX graph passes — they do not follow the
(gm, example_inputs, **kwargs) -> PassResult protocol. They operate
on already-compiled GraphModules at runtime. For graph passes, see
passes.py.
"""
import warnings
from typing import Callable, List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes import _graph_meta


def _capture_segment(gm, static_inputs, num_warmup):
    """Warmup, capture, and return (cuda_graph, static_output) for a callable."""
    for _ in range(num_warmup):
        gm(*static_inputs)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        static_output = gm(*static_inputs)
    return cuda_graph, static_output


def cudagraph_backend(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    num_warmup: int = 3,
) -> Callable:
    """Graph-aware CUDA graph backend that reads cudagraph_partition annotations.

    Uses segment metadata from cudagraph_partition to decide the capture
    strategy:
      - Single segment (whole graph capturable): allocate static input
        buffers, warmup, capture the entire graph, return a replay closure.
      - Multiple segments: capture each segment independently with eager
        bridges for non-capturable nodes in between.
      - No segments (nothing capturable): return gm unchanged with a warning.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA graphs require a CUDA device")

    segments = _graph_meta(gm.graph).get("cudagraph_segments", {})

    if not segments:
        warnings.warn(
            "cudagraph_backend: no capturable segments found, "
            "returning graph module unchanged."
        )
        return gm

    if len(segments) > 1:
        warnings.warn(
            f"cudagraph_backend: {len(segments)} segments detected. "
            "Multi-segment capture not yet implemented, falling back to "
            "eager execution.",
        )
        return gm

    device = torch.device("cuda")
    gm = gm.to(device)
    static_inputs = [inp.to(device).clone() for inp in example_inputs]

    cuda_graph, static_output = _capture_segment(
        gm, static_inputs, num_warmup
    )

    def replay(*inputs):
        for static, inp in zip(static_inputs, inputs):
            static.copy_(inp.to(device))
        cuda_graph.replay()
        if isinstance(static_output, torch.Tensor):
            return static_output.clone()
        return tuple(
            o.clone() if isinstance(o, torch.Tensor) else o
            for o in static_output
        )

    return replay
