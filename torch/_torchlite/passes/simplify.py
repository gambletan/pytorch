"""View chain simplification pass.

Compresses redundant view/reshape chains that arise from MHA decomposition
and other patterns. This reduces graph op count without changing semantics.
"""
from typing import List

import torch
from torch.fx import GraphModule, Node

from torch._torchlite.passes.common import (
    _set_phase,
    AddLayerNormKernel,
    AddRmsNormKernel,
    FusedKernel,
    MatmulEpilogueKernel,
    PassResult,
)


_CONTIGUOUS_PRODUCERS = frozenset({
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.zeros.default,
    torch.ops.aten.ones.default,
    torch.ops.aten.full.default,
})


def _is_contiguous_producer(node: Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if target in _CONTIGUOUS_PRODUCERS:
        return True
    if isinstance(target, (FusedKernel, MatmulEpilogueKernel, AddRmsNormKernel, AddLayerNormKernel)):
        return True
    return False


def _get_shape(node: Node):
    if not isinstance(node, Node):
        return None
    return node.meta.get("shape")


def simplify_views(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    graph = gm.graph
    changed = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        # --- Merge consecutive reshapes ---
        # reshape(reshape(x, s1), s2) -> reshape(x, s2)
        # when the intermediate reshape has only one user
        if node.target == torch.ops.aten.reshape.default:
            src = node.args[0]
            if (
                isinstance(src, Node)
                and src.op == "call_function"
                and src.target == torch.ops.aten.reshape.default
                and len(list(src.users)) == 1
            ):
                inner_src = src.args[0]
                node.replace_input_with(src, inner_src)
                graph.erase_node(src)
                changed = True
                # Fall through to check identity reshape on the merged result

            # --- Eliminate identity reshapes ---
            # reshape(x, shape) where shape == x.shape -> x
            # Only when target shape has no -1 (unresolved dims)
            actual_src = node.args[0]
            target_shape = list(node.args[1])
            src_shape = _get_shape(actual_src)
            if (
                src_shape is not None
                and -1 not in target_shape
                and list(src_shape) == target_shape
            ):
                node.replace_all_uses_with(actual_src)
                graph.erase_node(node)
                changed = True
                continue

        # --- Fold unsqueeze -> transpose -> squeeze into permute ---
        # unsqueeze(x, 0) -> transpose(0, K) -> squeeze(K)
        # becomes permute(x, [K-1, 0, 1, ..., K-2])
        if node.target == torch.ops.aten.unsqueeze.default:
            unsqueeze_dim = node.args[1]
            if unsqueeze_dim != 0:
                continue
            src = node.args[0]
            if not isinstance(src, Node):
                continue

            users = list(node.users.keys())
            if len(users) != 1:
                continue
            transpose_node = users[0]
            if (
                transpose_node.op != "call_function"
                or transpose_node.target != torch.ops.aten.transpose.int
            ):
                continue

            t_dim0, t_dim1 = transpose_node.args[1], transpose_node.args[2]
            # Normalize negative dims using the unsqueeze output's ndim
            # (src ndim + 1, since unsqueeze adds a dim at position 0)
            unsqueeze_ndim = len(_get_shape(src) or []) + 1
            if isinstance(t_dim0, int) and t_dim0 < 0:
                t_dim0 = t_dim0 + unsqueeze_ndim
            if isinstance(t_dim1, int) and t_dim1 < 0:
                t_dim1 = t_dim1 + unsqueeze_ndim
            if not (
                (t_dim0 == 0 and isinstance(t_dim1, int) and t_dim1 > 0)
                or (t_dim1 == 0 and isinstance(t_dim0, int) and t_dim0 > 0)
            ):
                continue
            k = t_dim1 if t_dim0 == 0 else t_dim0

            t_users = list(transpose_node.users.keys())
            if len(t_users) != 1:
                continue
            squeeze_node = t_users[0]
            if (
                squeeze_node.op != "call_function"
                or squeeze_node.target != torch.ops.aten.squeeze.dim
            ):
                continue
            squeeze_dim = squeeze_node.args[1]
            if isinstance(squeeze_dim, int) and squeeze_dim < 0:
                squeeze_dim = squeeze_dim + unsqueeze_ndim
            if squeeze_dim != k:
                continue

            # Build the permute dims: move dimension k-1 to front
            # Original x has ndim N. After unsqueeze(0) it's N+1.
            # transpose(0, K) swaps dims 0 and K. squeeze(K) removes dim K.
            # Net effect: dim K-1 of original moves to position 0.
            src_shape = _get_shape(src)
            if src_shape is None:
                continue
            ndim = len(src_shape)
            perm = [k - 1] + list(range(0, k - 1)) + list(range(k, ndim))

            out_shape = _get_shape(squeeze_node)
            phase = squeeze_node.meta.get("phase", "forward")

            graph.inserting_before(squeeze_node)
            permute_node = graph.call_function(
                torch.ops.aten.permute.default,
                (src, perm),
            )
            permute_node.name = squeeze_node.name
            if out_shape is not None:
                permute_node.meta["shape"] = list(out_shape)
            permute_node.meta["dtype"] = squeeze_node.meta.get("dtype")
            _set_phase(permute_node, phase)

            squeeze_node.replace_all_uses_with(permute_node)
            graph.erase_node(squeeze_node)
            graph.erase_node(transpose_node)
            graph.erase_node(node)
            changed = True
            continue

        # --- Remove clones feeding into matmul ops ---
        # clone -> (optional reshape) -> addmm/mm/bmm is redundant because
        # cuBLAS handles non-contiguous (strided) inputs natively via its
        # stride parameters, so the clone's only purpose (ensuring contiguity)
        # provides no benefit.
        if node.target == torch.ops.aten.clone.default:
            src = node.args[0]
            if isinstance(src, Node):
                _MATMUL_CONSUMERS = {
                    torch.ops.aten.addmm.default,
                    torch.ops.aten.mm.default,
                    torch.ops.aten.bmm.default,
                }
                all_users_are_matmul = True
                for user in node.users:
                    if user.op == "call_function" and user.target in _MATMUL_CONSUMERS:
                        continue
                    if (
                        user.op == "call_function"
                        and user.target == torch.ops.aten.reshape.default
                    ):
                        reshape_users = list(user.users.keys())
                        if all(
                            u.op == "call_function" and u.target in _MATMUL_CONSUMERS
                            for u in reshape_users
                        ):
                            continue
                    all_users_are_matmul = False
                    break

                if all_users_are_matmul and node.users:
                    node.replace_all_uses_with(src)
                    graph.erase_node(node)
                    changed = True
                    continue

        # --- Remove redundant clones ---
        # clone(x) where x is guaranteed contiguous -> x
        if node.target == torch.ops.aten.clone.default:
            src = node.args[0]
            if isinstance(src, Node) and _is_contiguous_producer(src):
                node.replace_all_uses_with(src)
                graph.erase_node(node)
                changed = True
                continue

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)
