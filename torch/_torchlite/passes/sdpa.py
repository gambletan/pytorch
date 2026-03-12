"""SDPA pattern matching pass: replace manual attention with F.scaled_dot_product_attention.

After decompose, manual multi-head attention produces a characteristic pattern:
  transpose(K) -> expand/reshape(Q,K) -> bmm(scores) -> view -> mul(scale) ->
  softmax -> expand/reshape(S,V) -> bmm(out) -> view
This pass detects that pattern and replaces it with a single SDPA call,
which dispatches to FlashAttention/memory-efficient kernels on GPU.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from torch._torchlite.passes.common import (
    _create_name,
    _set_phase,
    PassResult,
)


def _trace_through_views(node: Node) -> Node:
    """Walk backward through reshape/expand/_unsafe_view chains to find the source tensor."""
    view_ops = {
        torch.ops.aten.expand.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    }
    while (
        node.op == "call_function"
        and node.target in view_ops
        and isinstance(node.args[0], Node)
    ):
        node = node.args[0]
    return node


def _single_user(node: Node) -> Optional[Node]:
    """Return the sole user of node, or None if there isn't exactly one."""
    users = list(node.users.keys())
    if len(users) == 1:
        return users[0]
    return None


def _walk_forward_through_views(node: Node) -> Node:
    """Walk forward through single-use reshape/expand/_unsafe_view chains."""
    view_ops = {
        torch.ops.aten.expand.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    }
    while True:
        user = _single_user(node)
        if user is None:
            break
        if user.op != "call_function" or user.target not in view_ops:
            break
        node = user
    return node


def _infer_batch_size(q_node: Node) -> Optional[int]:
    """Infer the original batch dimension from the Q tensor's provenance.

    MHA decomposes Q as: select [S, B, E] → reshape [S, B*H, D] → transpose
    [B*H, S, D].  We walk backward through this chain to find B.  Returns
    None if the pattern doesn't match.
    """
    cur = q_node
    # Walk back through transpose (which swapped dims 0 and 1)
    if (
        cur.op == "call_function"
        and cur.target == torch.ops.aten.transpose.int
        and isinstance(cur.args[0], Node)
    ):
        cur = cur.args[0]
    # Now at reshape [S, B*H, D] — look for its input shape [S, B, E]
    if (
        cur.op == "call_function"
        and cur.target == torch.ops.aten.reshape.default
        and isinstance(cur.args[0], Node)
    ):
        src_shape = cur.args[0].meta.get("shape")
        if src_shape is not None and len(src_shape) >= 2:
            return src_shape[1]
    return None


def sdpa_pattern(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Replace manual bmm-scale-softmax-bmm attention with F.scaled_dot_product_attention."""
    graph = gm.graph

    # MHA with need_weights=True decomposes into attention weight averaging
    # (softmax → reshape → mean) that is dead when only output[0] is used.
    # Without DCE, softmax has 2 users and the pattern can't match.
    graph.eliminate_dead_code()

    changed = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (
            torch.ops.aten._softmax.default,
            torch.ops.aten._safe_softmax.default,
        ):
            continue

        # softmax(input, dim, half_to_float)
        # Check dim == -1
        if len(node.args) < 2:
            continue
        softmax_dim = node.args[1]
        if softmax_dim != -1 and softmax_dim != (len(node.meta.get("shape", [])) - 1):
            continue

        softmax_input = node.args[0]
        if not isinstance(softmax_input, Node):
            continue

        # Step 1: Check for scale op (mul or div) before softmax
        scale = None
        pre_scale_node = softmax_input
        if (
            softmax_input.op == "call_function"
            and softmax_input.target == torch.ops.aten.mul.Tensor
        ):
            arg0, arg1 = softmax_input.args[0], softmax_input.args[1]
            if isinstance(arg1, (int, float)):
                scale = float(arg1)
                pre_scale_node = arg0
            elif isinstance(arg0, (int, float)):
                scale = float(arg0)
                pre_scale_node = arg1
            else:
                continue
            if not isinstance(pre_scale_node, Node):
                continue
        elif (
            softmax_input.op == "call_function"
            and softmax_input.target == torch.ops.aten.div.Tensor
        ):
            arg0, arg1 = softmax_input.args[0], softmax_input.args[1]
            if isinstance(arg1, (int, float)) and isinstance(arg0, Node):
                scale = 1.0 / float(arg1)
                pre_scale_node = arg0
            else:
                continue
        else:
            # No scale op — softmax input goes directly to the view chain
            pre_scale_node = softmax_input

        # Step 2: Trace pre_scale_node back through _unsafe_view to find bmm1
        bmm1_output = _trace_through_views(pre_scale_node)
        if (
            bmm1_output.op != "call_function"
            or bmm1_output.target != torch.ops.aten.bmm.default
        ):
            continue

        bmm1 = bmm1_output

        # Step 3: Trace bmm1 inputs back through reshape/expand to find Q and K_transposed
        if len(bmm1.args) != 2:
            continue
        q_3d, kt_3d = bmm1.args[0], bmm1.args[1]
        if not isinstance(q_3d, Node) or not isinstance(kt_3d, Node):
            continue

        q_4d = _trace_through_views(q_3d)
        kt_4d = _trace_through_views(kt_3d)

        # MHA decomposes with the scale pre-applied to Q (and sometimes K)
        # before bmm, rather than scaling scores between bmm and softmax.
        # Detect this and extract the real Q/K so SDPA receives the correct
        # scale.  Handles both mul.Tensor and mul.Scalar.
        _MUL_OPS = (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        )
        q_pre_scale_node = None
        kt_pre_scale_node = None
        q_scale = None
        if (
            scale is None
            and q_4d.op == "call_function"
            and q_4d.target in _MUL_OPS
            and _single_user(q_4d) is not None
        ):
            a0, a1 = q_4d.args[0], q_4d.args[1]
            if isinstance(a1, (int, float)) and isinstance(a0, Node):
                q_scale = float(a1)
                q_pre_scale_node = q_4d
                q_4d = _trace_through_views(a0)
            elif isinstance(a0, (int, float)) and isinstance(a1, Node):
                q_scale = float(a0)
                q_pre_scale_node = q_4d
                q_4d = _trace_through_views(a1)

        # K may also be pre-scaled (need_weights=False decomposes with
        # both Q and K scaled by 1/sqrt(d_head)^0.5).
        kt_scale = None
        if (
            kt_4d.op == "call_function"
            and kt_4d.target in _MUL_OPS
            and _single_user(kt_4d) is not None
        ):
            a0, a1 = kt_4d.args[0], kt_4d.args[1]
            if isinstance(a1, (int, float)) and isinstance(a0, Node):
                kt_scale = float(a1)
                kt_pre_scale_node = kt_4d
                kt_4d = _trace_through_views(a0)
            elif isinstance(a0, (int, float)) and isinstance(a1, Node):
                kt_scale = float(a0)
                kt_pre_scale_node = kt_4d
                kt_4d = _trace_through_views(a1)

        # Compute the effective scale: scores = (Q*qs) @ (K*ks)^T = Q@K^T * qs*ks
        if q_scale is not None and kt_scale is not None:
            scale = q_scale * kt_scale
        elif q_scale is not None:
            scale = q_scale

        # Step 4: Check K is transposed: aten.transpose.int(K, -2, -1)
        if (
            kt_4d.op != "call_function"
            or kt_4d.target != torch.ops.aten.transpose.int
        ):
            continue
        if len(kt_4d.args) < 3:
            continue
        transpose_dims = (kt_4d.args[1], kt_4d.args[2])
        if transpose_dims != (-2, -1) and transpose_dims != (-1, -2):
            continue
        k_4d = kt_4d.args[0]
        if not isinstance(k_4d, Node):
            continue

        # Step 5: Trace softmax output forward through expand/reshape to find bmm2
        softmax_after_views = _walk_forward_through_views(node)
        bmm2_candidate = _single_user(softmax_after_views)
        if bmm2_candidate is None:
            continue
        if (
            bmm2_candidate.op != "call_function"
            or bmm2_candidate.target != torch.ops.aten.bmm.default
        ):
            continue

        bmm2 = bmm2_candidate

        # The softmax side should be arg[0] of bmm2, V side is arg[1]
        if len(bmm2.args) != 2:
            continue
        v_3d = bmm2.args[1]
        if not isinstance(v_3d, Node):
            continue
        v_4d = _trace_through_views(v_3d)

        # Step 6: Find the output _unsafe_view after bmm2
        output_node = _walk_forward_through_views(bmm2)

        # MHA with need_weights=False casts Q/K/V from fp16 to fp32 via
        # _to_copy before the attention computation.  SDPA handles mixed
        # precision internally (FlashAttention works on fp16 directly),
        # so trace through _to_copy to use the original fp16 tensors and
        # through _to_copy after bmm2 to find the fp16 output.
        cast_nodes = set()
        for label, ref in [("q", q_4d), ("k", k_4d), ("v", v_4d)]:
            if (
                ref.op == "call_function"
                and ref.target == torch.ops.aten._to_copy.default
                and isinstance(ref.args[0], Node)
                and _single_user(ref) is not None
            ):
                cast_nodes.add(ref)
                src = ref.args[0]
                if label == "q":
                    q_4d = src
                elif label == "k":
                    k_4d = src
                else:
                    v_4d = src

        # Also trace the output through a post-attention _to_copy (fp32→fp16)
        if cast_nodes:
            out_after = _walk_forward_through_views(output_node)
            user = _single_user(out_after)
            if (
                user is not None
                and user.op == "call_function"
                and user.target == torch.ops.aten._to_copy.default
            ):
                if out_after is not output_node:
                    cast_nodes.add(out_after)
                cast_nodes.add(user)
                output_node = user

        # Verify Q, K, V have the same dtype
        q_dtype = q_4d.meta.get("dtype")
        k_dtype = k_4d.meta.get("dtype")
        v_dtype = v_4d.meta.get("dtype")
        if q_dtype is None or q_dtype != k_dtype or q_dtype != v_dtype:
            continue

        # For fp32 with small sequence lengths (<=32), memory-efficient
        # attention (fmha_cutlassF) has higher overhead than the decomposed
        # bmm+softmax+bmm path. The flash/mem-eff kernels are optimized for
        # long sequences; for short ones the launch overhead dominates.
        q_shape_check = q_4d.meta.get("shape")
        if q_dtype == torch.float32 and q_shape_check is not None:
            seq_dim = -2
            seq_len = q_shape_check[seq_dim] if len(q_shape_check) >= 2 else None
            if seq_len is not None and seq_len <= 32:
                continue

        # Verify all intermediate nodes in the matched subgraph are single-use
        # (Q, K, V themselves may have other users)
        intermediate_nodes = set()
        # Collect all nodes between Q/K/V and the output
        intermediate_nodes.add(kt_4d)  # transpose
        if q_pre_scale_node is not None:
            intermediate_nodes.add(q_pre_scale_node)
        if kt_pre_scale_node is not None:
            intermediate_nodes.add(kt_pre_scale_node)
        intermediate_nodes.update(cast_nodes)
        # Forward chain from Q to bmm1
        _collect_view_chain(q_4d, bmm1, intermediate_nodes)
        _collect_view_chain(kt_4d, bmm1, intermediate_nodes)
        intermediate_nodes.add(bmm1)
        # bmm1 output to softmax
        _collect_view_chain(bmm1, node, intermediate_nodes)
        if scale is not None:
            intermediate_nodes.add(softmax_input)  # the mul/div node
        intermediate_nodes.add(node)  # softmax
        # softmax to bmm2
        _collect_view_chain(node, bmm2, intermediate_nodes)
        _collect_view_chain(v_4d, bmm2, intermediate_nodes)
        intermediate_nodes.add(bmm2)
        # bmm2 to output
        _collect_view_chain(bmm2, output_node, intermediate_nodes)
        if output_node is not bmm2:
            intermediate_nodes.add(output_node)

        # Check single-use for intermediates (excluding Q, K, V, and the output node)
        skip_check = {q_4d, k_4d, v_4d, output_node}
        single_use_ok = True
        for n in intermediate_nodes:
            if n in skip_check:
                continue
            if len(list(n.users.keys())) > 1:
                single_use_ok = False
                break
        if not single_use_ok:
            continue

        # Step 7: Replace with SDPA
        output_shape = output_node.meta.get("shape")
        phase = output_node.meta.get("phase", "forward")

        graph.inserting_before(output_node)
        sdpa_kwargs = {}
        if scale is not None:
            sdpa_kwargs["scale"] = scale

        # FlashAttention requires 4D [B, H, S, D] input.  MHA decomposition
        # produces 3D [B*H, S, D] — detect this and insert reshapes so the
        # SDPA dispatch picks the flash kernel instead of naive bmm.
        q_sdpa, k_sdpa, v_sdpa = q_4d, k_4d, v_4d
        need_3d_fixup = False
        q_shape = q_4d.meta.get("shape")
        if q_shape is not None and len(q_shape) == 3:
            batch_size = _infer_batch_size(q_4d)
            if batch_size is not None and q_shape[0] % batch_size == 0:
                nhead = q_shape[0] // batch_size
                seq_len, d_head = q_shape[1], q_shape[2]
                shape_4d = [batch_size, nhead, seq_len, d_head]
                need_3d_fixup = True
                for name_suffix, src in [("q", q_4d), ("k", k_4d), ("v", v_4d)]:
                    rn = graph.call_function(
                        torch.ops.aten.reshape.default,
                        (src, shape_4d),
                    )
                    rn.name = _create_name(graph, f"sdpa_{name_suffix}_4d")
                    rn.meta["shape"] = shape_4d
                    rn.meta["dtype"] = q_dtype
                    _set_phase(rn, phase)
                    if name_suffix == "q":
                        q_sdpa = rn
                    elif name_suffix == "k":
                        k_sdpa = rn
                    else:
                        v_sdpa = rn

        sdpa_node = graph.call_function(
            F.scaled_dot_product_attention,
            (q_sdpa, k_sdpa, v_sdpa),
            sdpa_kwargs,
        )
        sdpa_node.name = _create_name(graph, "sdpa")
        _set_phase(sdpa_node, phase)
        if need_3d_fixup:
            sdpa_node.meta["shape"] = [batch_size, nhead, seq_len, d_head]
            # Reshape back to 3D [B*H, S, D] for downstream compatibility
            reshape_back = graph.call_function(
                torch.ops.aten.reshape.default,
                (sdpa_node, list(q_shape)),
            )
            reshape_back.name = _create_name(graph, "sdpa_3d")
            reshape_back.meta["shape"] = list(q_shape)
            reshape_back.meta["dtype"] = q_dtype
            _set_phase(reshape_back, phase)
            output_node.replace_all_uses_with(reshape_back)
        else:
            if output_shape is not None:
                sdpa_node.meta["shape"] = output_shape
            output_node.replace_all_uses_with(sdpa_node)
        sdpa_node.meta["dtype"] = q_dtype

        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)


def _collect_view_chain(start: Node, end: Node, collected: set) -> None:
    """Collect view nodes on the path from start to end (exclusive of start, inclusive of intermediates)."""
    view_ops = {
        torch.ops.aten.expand.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    }
    current = start
    while current is not end:
        user = _single_user(current)
        if user is None or user.op != "call_function":
            break
        if user is end:
            break
        if user.target in view_ops or user.target in (
            torch.ops.aten.bmm.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten._softmax.default,
        ):
            collected.add(user)
            current = user
        else:
            break


def _erase_dead_nodes(graph, candidates: set) -> None:
    """Erase nodes from candidates set that have no users, in reverse topological order."""
    # Build reverse topological order from graph
    topo = list(graph.nodes)
    for node in reversed(topo):
        if node in candidates and not node.users:
            graph.erase_node(node)
