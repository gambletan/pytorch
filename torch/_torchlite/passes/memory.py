"""Memory planning pass."""
from typing import Dict, List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _graph_meta,
    PassResult,
)


_VIEW_OPS = frozenset({
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.permute.default,
    torch.ops.aten._unsafe_view.default,
})


def memory_plan(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Perform liveness analysis and assign pool IDs to tensor-producing nodes.

    This is an analysis-only pass: it computes live ranges and greedily assigns
    pool IDs as node metadata (node.meta["memory_pool"]), but does NOT implement
    runtime buffer reuse. No actual memory is saved at execution time — pool IDs
    are informational and intended for downstream tooling or future allocators.
    """
    graph = gm.graph

    node_order = {}
    for i, node in enumerate(graph.nodes):
        node_order[node] = i

    # Compute liveness: for each tensor-producing call_function node,
    # record (creation_time, last_use_time, size_in_bytes).
    # Parameters and inputs are pre-allocated and excluded.
    # View ops share storage with their source tensor, so we must
    # transitively follow view users to find the true last consumer
    # of the underlying memory.
    def _transitive_last_use(node, node_order):
        best = node_order[node]
        worklist = list(node.users)
        while worklist:
            user = worklist.pop()
            t = node_order.get(user, best)
            if t > best:
                best = t
            if user.op == "call_function" and user.target in _VIEW_OPS:
                worklist.extend(user.users)
        return best

    intervals: Dict[torch.fx.Node, tuple] = {}
    for node in graph.nodes:
        if node.op not in ("call_function", "call_module"):
            continue
        if node.op == "call_function" and node.target in _VIEW_OPS:
            continue
        shape = node.meta.get("shape")
        if shape is None:
            continue

        creation = node_order[node]
        last_use = _transitive_last_use(node, node_order)

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

    # Find the output node's source tensor (trace through views). Exclude
    # it from pool assignment so it allocates a fresh tensor each call,
    # avoiding the clone() on the return value.
    output_source = None
    for node in graph.nodes:
        if node.op == "output":
            out_val = node.args[0]
            if isinstance(out_val, (tuple, list)):
                out_val = out_val[0]
            while (
                hasattr(out_val, "op")
                and out_val.op == "call_function"
                and out_val.target in _VIEW_OPS
            ):
                out_val = out_val.args[0]
            if hasattr(out_val, "op") and out_val in intervals:
                output_source = out_val
            break

    # Greedy best-fit pool assignment: reuse the smallest free pool that
    # can hold the new tensor.  Pools track dtype to prevent aliasing
    # buffers of different element types (e.g. float16 vs float32).
    # pools[i] = [capacity_bytes, free_after_time, dtype].
    pools: List[list] = []
    assignments: Dict[torch.fx.Node, int] = {}

    for node in sorted(intervals, key=lambda n: intervals[n][0]):
        if node is output_source:
            continue
        creation, last_use, size = intervals[node]
        dtype = node.meta.get("dtype", torch.float32)

        best = None
        best_cap = float("inf")
        for i, (cap, free_after, pool_dtype) in enumerate(pools):
            if (
                pool_dtype == dtype
                and free_after < creation
                and cap >= size
                and cap < best_cap
            ):
                best = i
                best_cap = cap

        if best is not None:
            pools[best][1] = last_use
            assignments[node] = best
        else:
            assignments[node] = len(pools)
            pools.append([size, last_use, dtype])

    naive_alloc = sum(sz for _, _, sz in intervals.values())
    planned_alloc = sum(p[0] for p in pools)

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
