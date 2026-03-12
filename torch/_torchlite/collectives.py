"""Collective communication ops and DTensor placement propagation.

Provides callable collective ops (allgather, reduce_scatter, allreduce) that
appear as nodes in the FX graph, plus placement propagation helpers used by
the subclass_unwrap pass to determine how DTensor placements flow through
each operation in the graph.
"""
import logging
import warnings

import torch

log = logging.getLogger(__name__)

from torch._torchlite.ops import _named, _UNARY_POINTWISE_OPS


def _try_real_collective(name, tensor, mesh=None):
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            from torch.distributed import functional_collectives as funcol
            group = mesh if mesh is not None else dist.group.WORLD
            if name == "allgather":
                return funcol.all_gather_tensor(tensor, 0, group)
            elif name == "reduce_scatter":
                return funcol.reduce_scatter_tensor(tensor, "sum", 0, group)
            elif name == "allreduce":
                return funcol.all_reduce(tensor, "sum", group)
        else:
            warnings.warn(
                f"Collective '{name}' called but torch.distributed is not "
                f"initialized. Returning tensor unchanged (identity fallback).",
                stacklevel=3,
            )
    except (RuntimeError, ValueError) as e:
        # Only fall back to identity for single-rank groups (e.g. fake
        # process groups used in tests). Multi-rank failures are real
        # errors that must propagate.
        try:
            is_multi_rank = dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            raise e
        if is_multi_rank:
            raise
        log.debug(
            "Collective '%s' failed in single-rank group, "
            "identity fallback: %s",
            name, e,
        )
    return tensor


@_named("allgather")
def _allgather(tensor):
    return _try_real_collective("allgather", tensor)


@_named("reduce_scatter")
def _reduce_scatter(tensor):
    return _try_real_collective("reduce_scatter", tensor)


@_named("allreduce")
def _allreduce(tensor):
    return _try_real_collective("allreduce", tensor)


# ── Placement propagation helpers ────────────────────────────────────────────

# Ops whose DTensor placement propagates transparently (output placement
# matches input placement). See also _POINTWISE_OPS in passes.py which
# lists ops eligible for Triton fusion — the two sets overlap but are
# intentionally separate since they serve different purposes.
_POINTWISE_PLACEMENT_OPS = _UNARY_POINTWISE_OPS | frozenset({
    "dropout", "gelu", "silu", "div", "clone", "contiguous",
})

_BINARY_ELEMENTWISE_OPS = frozenset({
    "add", "sub", "mul", "pow",
})

_MATMUL_OPS = frozenset({"matmul", "mm"})

_REDUCTION_OPS = frozenset({"mean", "sum", "softmax"})

_VIEW_OPS = frozenset({"transpose", "permute", "reshape", "view"})

_NORM_OPS = frozenset({"layer_norm", "rms_norm"})


def _is_shard(spec):
    return spec is not None and spec[0] == "Shard"


def _is_replicate(spec):
    return spec is not None and spec[0] == "Replicate"


def _propagate_pointwise(node, placement):
    for a in node.args:
        if isinstance(a, torch.fx.Node) and a in placement:
            spec = placement[a]
            if spec[0] == "_Partial":
                spec = ("Replicate", None)
            placement[node] = spec
            return True
    return False


def _propagate_binary(node, placement):
    tensor_args = [
        (a, placement.get(a))
        for a in node.args
        if isinstance(a, torch.fx.Node)
    ]
    if len(tensor_args) >= 2:
        a_spec = tensor_args[0][1]
        b_spec = tensor_args[1][1]
        if a_spec and a_spec[0] == "_Partial":
            a_spec = ("Replicate", None)
        if b_spec and b_spec[0] == "_Partial":
            b_spec = ("Replicate", None)
        # Both sharded on different dims is unsupported — requires a
        # redistribution that we can't insert automatically. Warn and
        # fall back to treating the result as Replicate.
        if (
            _is_shard(a_spec) and _is_shard(b_spec)
            and a_spec[1] != b_spec[1]
        ):
            warnings.warn(
                f"Binary op '{node.name}': operands sharded on different "
                f"dims ({a_spec}, {b_spec}). Treating result as Replicate."
            )
            placement[node] = ("Replicate", None)
        elif _is_shard(a_spec):
            placement[node] = a_spec
        elif _is_shard(b_spec):
            placement[node] = b_spec
        elif a_spec:
            placement[node] = a_spec
        elif b_spec:
            placement[node] = b_spec
        else:
            return False
        return True
    elif len(tensor_args) == 1 and tensor_args[0][1]:
        spec = tensor_args[0][1]
        if spec[0] == "_Partial":
            spec = ("Replicate", None)
        placement[node] = spec
        return True
    return False


def _propagate_matmul(node, placement):
    a = node.args[0] if len(node.args) > 0 else None
    b = node.args[1] if len(node.args) > 1 else None
    a_spec = placement.get(a) if isinstance(a, torch.fx.Node) else None
    b_spec = placement.get(b) if isinstance(b, torch.fx.Node) else None

    # For matmul/mm the contraction is on a's last dim and b's second-to-last
    # dim. For 2D this is a[1] and b[0]; for batched matmul (3D+) we need to
    # compute the actual indices from the operand shapes.
    a_shape = a.meta.get("shape") if isinstance(a, torch.fx.Node) else None
    b_shape = b.meta.get("shape") if isinstance(b, torch.fx.Node) else None
    a_contract = (len(a_shape) - 1) if a_shape else 1
    b_contract = (len(b_shape) - 2) if b_shape and len(b_shape) >= 2 else 0

    if (
        _is_shard(a_spec) and a_spec[1] == a_contract
        and _is_shard(b_spec) and b_spec[1] == b_contract
    ):
        placement[node] = ("_Partial", None)
    elif _is_replicate(a_spec) and _is_shard(b_spec):
        placement[node] = b_spec
    elif _is_shard(a_spec) and _is_replicate(b_spec):
        placement[node] = a_spec
    elif _is_replicate(a_spec) and _is_replicate(b_spec):
        placement[node] = ("Replicate", None)
    else:
        if isinstance(b, torch.fx.Node) and b in placement:
            placement[node] = placement[b]
            return True
        elif isinstance(a, torch.fx.Node) and a in placement:
            placement[node] = placement[a]
            return True
        return False
    return True


def _propagate_reduction(node, placement):
    a = node.args[0] if node.args else None
    if not isinstance(a, torch.fx.Node) or a not in placement:
        return False

    spec = placement[a]
    red_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
    keepdim = node.kwargs.get("keepdim", False)
    if len(node.args) > 2 and isinstance(node.args[2], bool):
        keepdim = node.args[2]

    if not _is_shard(spec) or red_dim is None:
        placement[node] = ("Replicate", None)
        return True

    shard_dim = spec[1]
    if isinstance(red_dim, (list, tuple)):
        if shard_dim in red_dim:
            placement[node] = ("Replicate", None)
        else:
            new_dim = shard_dim
            if not keepdim:
                new_dim -= sum(1 for d in red_dim if d < shard_dim)
            placement[node] = ("Shard", new_dim)
    elif red_dim == shard_dim:
        placement[node] = ("Replicate", None)
    else:
        new_dim = shard_dim
        if not keepdim and red_dim < shard_dim:
            new_dim -= 1
        placement[node] = ("Shard", new_dim)
    return True


def _propagate_view(node, placement):
    a = node.args[0] if node.args else None
    if not isinstance(a, torch.fx.Node) or a not in placement:
        return False

    spec = placement[a]
    name = getattr(node.target, "__name__", "")

    if name == "transpose" and _is_shard(spec):
        d0 = node.args[1] if len(node.args) > 1 else 0
        d1 = node.args[2] if len(node.args) > 2 else 1
        shard_dim = spec[1]
        if shard_dim == d0:
            placement[node] = ("Shard", d1)
        elif shard_dim == d1:
            placement[node] = ("Shard", d0)
        else:
            placement[node] = spec
    elif name == "permute" and _is_shard(spec):
        perm = node.args[1] if len(node.args) > 1 else None
        if isinstance(perm, (list, tuple)):
            shard_dim = spec[1]
            try:
                new_dim = list(perm).index(shard_dim)
                placement[node] = ("Shard", new_dim)
            except ValueError:
                placement[node] = ("Replicate", None)
        else:
            placement[node] = spec
    else:
        placement[node] = spec
    return True


def _propagate_norm(node, placement):
    a = node.args[0] if node.args else None
    if isinstance(a, torch.fx.Node) and a in placement:
        placement[node] = placement[a]
        return True
    return False


def _propagate_embedding(node, placement):
    # embedding(weight, indices) — if weight is Shard(0) on vocab dim,
    # output requires allgather (handled by collective insertion later).
    # Propagate the input (indices) placement to the output.
    if len(node.args) >= 2:
        weight = node.args[0]
        indices = node.args[1]
        w_spec = placement.get(weight) if isinstance(weight, torch.fx.Node) else None
        i_spec = placement.get(indices) if isinstance(indices, torch.fx.Node) else None
        if _is_shard(w_spec) and w_spec[1] == 0:
            placement[node] = ("Replicate", None)
            return True
        if i_spec:
            placement[node] = i_spec
            return True
    return False


def _has_dtensor_params(gm):
    for node in gm.graph.nodes:
        if node.op in ("get_attr", "placeholder"):
            if node.meta.get("dtensor_spec") is not None:
                return True
    return False
