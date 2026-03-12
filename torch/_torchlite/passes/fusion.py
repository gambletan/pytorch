"""Fusion pass: fuse chains of pointwise ops into FusedKernel nodes."""
from typing import Dict, List

import torch
from torch.fx import GraphModule

import operator

import torch.nn.functional as _F

from torch._torchlite.passes.common import (
    _aten_op_name,
    _create_name,
    _graph_meta,
    _node_shape,
    _set_phase,
    AddLayerNormKernel,
    AddRmsNormKernel,
    FusedKernel,
    FusedOp,
    FusionGroup,
    MatmulEpilogueKernel,
    PassResult,
)
from torch._torchlite.ops import _UNARY_POINTWISE_OPS

_POINTWISE_OPS = _UNARY_POINTWISE_OPS | frozenset({
    "add", "sub", "mul", "div", "where",
})


def fuse(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Fuse chains of pointwise ops into FusedKernel nodes.

    WARNING: After this pass, the graph is no longer directly executable.
    FusedKernel nodes are placeholders that require triton_codegen and
    precompile passes, followed by precompile_save/precompile_load to
    produce a runnable module.
    """
    graph = gm.graph
    groups: List[FusionGroup] = []
    node_to_group: Dict[torch.fx.Node, FusionGroup] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        op_name = _aten_op_name(node.target)
        if op_name not in _POINTWISE_OPS:
            continue
        if node.kwargs:
            continue

        shape = _node_shape(node)

        merged = False
        for arg in node.args:
            if not isinstance(arg, torch.fx.Node) or arg not in node_to_group:
                continue
            # Don't merge through a node that has users outside the pointwise
            # universe — those users need the intermediate value to survive,
            # so the node can't be absorbed into a fused kernel.
            has_blocking_user = False
            for user in arg.users:
                if user is node:
                    continue
                if user.op != "call_function":
                    has_blocking_user = True
                    break
                if _aten_op_name(user.target) not in _POINTWISE_OPS:
                    has_blocking_user = True
                    break
            if has_blocking_user:
                continue
            group = node_to_group[arg]
            if group.shape is not None and group.shape == shape:
                group.node_names.append(node.name)
                group.op_names.append(op_name)
                group.output = node.name
                node_to_group[node] = group
                merged = True
                break

        if not merged:
            group = FusionGroup(
                group_id=len(groups),
                node_names=[node.name],
                op_names=[op_name],
                shape=shape,
                inputs=[],
                output=node.name,
            )
            groups.append(group)
            node_to_group[node] = group

    for group in groups:
        group_set = set(group.node_names)
        inputs = set()
        for n in graph.nodes:
            if n.name not in group_set:
                continue
            for arg in n.args:
                if isinstance(arg, torch.fx.Node) and arg.name not in group_set:
                    inputs.add(arg.name)
        group.inputs = sorted(inputs)

    multi_op = [g for g in groups if len(g.node_names) >= 2]

    name_to_node = {n.name: n for n in graph.nodes}
    counter = 0
    replaced = {}

    for group in multi_op:
        group.inputs = [replaced.get(inp, inp) for inp in group.inputs]
        group_set = set(group.node_names)

        input_index = {inp: i for i, inp in enumerate(group.inputs)}
        tmp_index: Dict[str, int] = {}

        fused_ops = []
        for idx, (node_name, op_name) in enumerate(
            zip(group.node_names, group.op_names)
        ):
            node = name_to_node[node_name]
            args = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name in input_index:
                        args.append(("input", input_index[arg.name]))
                    elif arg.name in tmp_index:
                        args.append(("tmp", tmp_index[arg.name]))
                else:
                    args.append(("const", arg))
            fused_ops.append(FusedOp(op_name=op_name, args=args))
            tmp_index[node_name] = idx

        kernel_name = "fused_" + "_".join(group.op_names[:4])
        if len(group.op_names) > 4:
            kernel_name += f"_x{len(group.op_names)}"
        kernel_name = f"{kernel_name}_{counter}"
        counter += 1

        input_nodes = [name_to_node[inp] for inp in group.inputs]
        input_shapes = [_node_shape(n) for n in input_nodes]

        kernel = FusedKernel(
            name=kernel_name,
            ops=fused_ops,
            n_inputs=len(group.inputs),
            shape=group.shape,
            input_shapes=input_shapes,
        )
        output_node = name_to_node[group.output]

        graph.inserting_before(output_node)
        fused_node = graph.call_function(kernel, tuple(input_nodes))
        fused_node.name = _create_name(graph, kernel_name)
        _set_phase(fused_node, output_node.meta.get("phase", "forward"))
        if group.shape:
            fused_node.meta["shape"] = group.shape
        fused_node.meta["dtype"] = output_node.meta.get("dtype", torch.float32)

        output_node.replace_all_uses_with(fused_node)
        replaced[group.output] = fused_node.name
        name_to_node[fused_node.name] = fused_node

        for node_name in reversed(group.node_names):
            node = name_to_node[node_name]
            if not node.users:
                del name_to_node[node_name]
                graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def matmul_epilogue(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Fuse addmm/mm followed by pointwise chains into MatmulEpilogueKernel nodes.

    Pattern-matches sequences like addmm -> relu -> mul and replaces
    them with a single node whose target is a MatmulEpilogueKernel.
    The triton_lower pass later compiles these into 2D tiled Triton
    kernels that compute the matmul and epilogue in one launch.

    For 3D inputs (e.g. [B, S, K] shaped tensors), PyTorch's linear
    decomposition emits view([B*S, K]) -> mm -> _unsafe_view([B, S, N]).
    We look through the _unsafe_view so that epilogue ops operating on
    the restored 3D shape (e.g. silu, mul) are still fused.  The mm node
    itself always operates on the 2D [B*S, N] result, so M = B*S remains
    the batch dimension passed to the Triton kernel.
    """
    graph = gm.graph
    changed = False
    counter = 0

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        target = node.target

        is_addmm = target == torch.ops.aten.addmm.default
        is_mm = target == torch.ops.aten.mm.default
        if not is_addmm and not is_mm:
            continue

        mm_shape = _node_shape(node)
        if mm_shape is None or len(mm_shape) != 2:
            continue
        M, N = mm_shape

        # When the input is 3D ([B, S, K]), decompose emits:
        #   view([B*S, K]) -> mm([B*S, N]) -> _unsafe_view([B, S, N])
        # The epilogue pointwise chain starts from _unsafe_view, not mm.
        # Detect this reshape-to-batch pattern and treat the _unsafe_view
        # as a transparent alias so the epilogue chain is still fused.
        reshape_node = None
        reshape_shape = None
        mm_users = list(node.users.keys())
        # For fp32 we also look through reshape/view since the Triton
        # matmul kernel is competitive with cuBLAS and fusing the epilogue
        # saves kernel launches. For fp16, cuBLAS tensor-core matmul is
        # much faster and the epilogue fallback applies ops separately,
        # so we only look through _unsafe_view (the decomposition output).
        node_dtype = node.meta.get("dtype", torch.float32)
        if node_dtype in (torch.float32, torch.bfloat16):
            _reshape_targets = (
                torch.ops.aten._unsafe_view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.view.default,
            )
        else:
            _reshape_targets = (torch.ops.aten._unsafe_view.default,)
        if (
            len(mm_users) == 1
            and mm_users[0].op == "call_function"
            and mm_users[0].target in _reshape_targets
        ):
            candidate = mm_users[0]
            cshape = _node_shape(candidate)
            if cshape is not None and len(cshape) > 2:
                reshaped_n = cshape[-1]
                reshaped_m = 1
                for s in cshape[:-1]:
                    reshaped_m *= s
                if reshaped_m == M and reshaped_n == N:
                    reshape_node = candidate
                    reshape_shape = cshape

        mm_nodes = {node.name}
        if reshape_node is not None:
            mm_nodes.add(reshape_node.name)

        # The effective M×N for epilogue extra-arg shape validation:
        # after the _unsafe_view the tensors are back in 3D space, so
        # we track ep_shape as the actual output shape of the chain root.
        ep_root = reshape_node if reshape_node is not None else node
        ep_shape = reshape_shape if reshape_shape is not None else mm_shape
        ep_n = ep_shape[-1]

        chain = []
        current = ep_root
        while True:
            users = list(current.users.keys())
            if len(users) != 1:
                break
            user = users[0]
            if user.op != "call_function":
                break
            user_op = _aten_op_name(user.target)
            if user_op not in _POINTWISE_OPS:
                break
            if user.kwargs:
                break
            has_bad_extra = False
            for arg in user.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                if arg.name in mm_nodes or arg.name in {c.name for c in chain}:
                    continue
                arg_shape = _node_shape(arg)
                if arg_shape is None:
                    has_bad_extra = True
                    break
                # Allow scalars, 1D broadcast vectors of length N, and
                # tensors whose shape matches the current epilogue shape.
                if len(arg_shape) == 0:
                    pass
                elif len(arg_shape) == 1 and arg_shape[0] == ep_n:
                    pass
                elif list(arg_shape) == list(ep_shape):
                    pass
                else:
                    has_bad_extra = True
                    break
            if has_bad_extra:
                break
            chain.append(user)
            mm_nodes.add(user.name)
            current = user

        # For fp16 with a reshape between mm and epilogue, cuBLAS
        # almost always wins the matmul benchmark and the epilogue ops
        # are applied as separate kernels. The fuse pass creates more
        # efficient fused Triton kernels for these chains, so skip the
        # epilogue absorption when going through a reshape for fp16.
        if node_dtype == torch.float16 and reshape_node is not None:
            chain = []

        if not chain:
            continue

        if is_addmm:
            bias_node, input_node, weight_t_node = node.args[0], node.args[1], node.args[2]
        else:
            input_node, weight_t_node = node.args[0], node.args[1]
            bias_node = None

        input_shape = _node_shape(input_node)
        if input_shape is None or len(input_shape) != 2:
            continue
        K = input_shape[1]

        external_inputs = [input_node, weight_t_node]
        if bias_node is not None:
            external_inputs.append(bias_node)
        ext_input_names = {n.name for n in external_inputs}

        extra_consts = []
        extra_shapes = []
        epilogue_ops = []
        group_names = {node.name} | {c.name for c in chain}

        # Names that map to the accumulator: the raw mm output and the
        # optional _unsafe_view that restores the original batch shape.
        acc_names = {node.name}
        if reshape_node is not None:
            acc_names.add(reshape_node.name)

        for ep_node in chain:
            ep_op = _aten_op_name(ep_node.target)
            args = []
            for arg in ep_node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name in acc_names:
                        args.append(("acc", 0))
                    elif arg.name in ext_input_names:
                        idx = next(
                            i for i, n in enumerate(external_inputs)
                            if n.name == arg.name
                        )
                        args.append(("input", idx))
                    elif arg.name in {c.name for c in chain}:
                        idx = next(
                            i for i, c in enumerate(chain)
                            if c.name == arg.name
                        )
                        args.append(("tmp", idx))
                    else:
                        ci = len(extra_consts)
                        extra_consts.append(arg)
                        extra_shapes.append(tuple(_node_shape(arg) or ()))
                        args.append(("extra", ci))
                else:
                    args.append(("const", arg))
            epilogue_ops.append(FusedOp(op_name=ep_op, args=args))

        kernel_name = f"matmul_epilogue_{'_'.join(op.op_name for op in epilogue_ops[:3])}_{counter}"
        counter += 1

        dtype = node.meta.get("dtype", torch.float32)
        # The fused kernel output shape is the epilogue chain's output shape,
        # which is 3D when reshape_node is present.
        out_shape = _node_shape(chain[-1]) or ep_shape
        kernel = MatmulEpilogueKernel(
            name=kernel_name,
            epilogue_ops=epilogue_ops,
            has_bias=bias_node is not None,
            M=M, N=N, K=K,
            dtype=dtype,
            extra_shapes=extra_shapes,
            out_shape=list(out_shape),
        )

        all_inputs = tuple(external_inputs) + tuple(extra_consts)
        last_ep = chain[-1]
        graph.inserting_before(last_ep)
        fused_node = graph.call_function(kernel, all_inputs)
        fused_node.name = _create_name(graph, kernel_name)
        _set_phase(fused_node, last_ep.meta.get("phase", "forward"))
        fused_node.meta["shape"] = list(out_shape)
        fused_node.meta["dtype"] = dtype

        last_ep.replace_all_uses_with(fused_node)

        for ep_node in reversed(chain):
            if not ep_node.users:
                graph.erase_node(ep_node)
        if reshape_node is not None and not reshape_node.users:
            graph.erase_node(reshape_node)
        if not node.users:
            graph.erase_node(node)

        changed = True

    if changed:
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)


def fuse_add_rms_norm(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Fuse add + rms_norm and standalone rms_norm into AddRmsNormKernel nodes.

    Matches two patterns:
    1. c = add(a, b); out = rms_norm(c, shape, weight, eps) — fused add+norm
       producing (add_result, norm_result) tuple via getitem.
    2. out = rms_norm(x, shape, weight, eps) — standalone norm (has_add=False)
       producing a single norm result, dispatched as a Triton reduction kernel
       instead of the standalone CUDA rms_norm kernel.
    """
    graph = gm.graph
    changed = False
    counter = 0

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _F.rms_norm:
            continue

        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue

        shape = _node_shape(node)
        if shape is None or len(shape) < 2:
            continue

        norm_dim = shape[-1]
        normalized_shape = node.args[1] if len(node.args) > 1 else None
        if normalized_shape is None or normalized_shape != (norm_dim,):
            continue

        weight_node = node.kwargs.get("weight")
        if weight_node is None or not isinstance(weight_node, torch.fx.Node):
            continue

        eps = node.kwargs.get("eps")
        if eps is None:
            eps = 1e-6

        dtype = node.meta.get("dtype", torch.float32)

        has_add = (
            input_node.op == "call_function"
            and _aten_op_name(input_node.target) == "add"
        )

        if has_add:
            kernel_name = f"add_rms_norm_{counter}"
        else:
            kernel_name = f"rms_norm_{counter}"
        counter += 1

        kernel = AddRmsNormKernel(
            name=kernel_name,
            shape=list(shape),
            norm_dim=norm_dim,
            eps=eps,
            dtype=dtype,
            has_add=has_add,
        )

        if has_add:
            add_node = input_node
            a_node, b_node = add_node.args[0], add_node.args[1]

            graph.inserting_before(node)
            fused_node = graph.call_function(
                kernel, (a_node, b_node, weight_node)
            )
            fused_node.name = _create_name(graph, kernel_name)
            _set_phase(fused_node, node.meta.get("phase", "forward"))
            fused_node.meta["shape"] = list(shape)
            fused_node.meta["dtype"] = dtype

            add_getitem = graph.call_function(operator.getitem, (fused_node, 0))
            add_getitem.name = _create_name(graph, f"{kernel_name}_add")
            add_getitem.meta["shape"] = list(shape)
            add_getitem.meta["dtype"] = dtype

            norm_getitem = graph.call_function(operator.getitem, (fused_node, 1))
            norm_getitem.name = _create_name(graph, f"{kernel_name}_norm")
            norm_getitem.meta["shape"] = list(shape)
            norm_getitem.meta["dtype"] = dtype

            node.replace_all_uses_with(norm_getitem)
            add_node.replace_all_uses_with(add_getitem)
            add_getitem.replace_input_with(add_getitem, fused_node)

            if not node.users:
                graph.erase_node(node)
            if not add_node.users:
                graph.erase_node(add_node)
        else:
            graph.inserting_before(node)
            fused_node = graph.call_function(
                kernel, (input_node, weight_node)
            )
            fused_node.name = _create_name(graph, kernel_name)
            _set_phase(fused_node, node.meta.get("phase", "forward"))
            fused_node.meta["shape"] = list(shape)
            fused_node.meta["dtype"] = dtype

            node.replace_all_uses_with(fused_node)
            if not node.users:
                graph.erase_node(node)

        changed = True

    if changed:
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)


def fuse_add_layer_norm(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> PassResult:
    """Fuse add + layer_norm and standalone layer_norm into AddLayerNormKernel nodes.

    Matches two patterns:
    1. c = add(a, b); out = layer_norm(c, shape, weight, bias, eps) — fused
       add+norm producing (add_result, norm_result) tuple via getitem.
    2. out = layer_norm(x, shape, weight, bias, eps) — standalone norm
       (has_add=False) producing a single norm result, dispatched as a Triton
       reduction kernel instead of the standalone CUDA layer_norm kernel.
    """
    graph = gm.graph
    changed = False
    counter = 0

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _F.layer_norm:
            continue

        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue

        shape = _node_shape(node)
        if shape is None or len(shape) < 2:
            continue

        norm_dim = shape[-1]
        normalized_shape = node.args[1] if len(node.args) > 1 else None
        if normalized_shape is None or list(normalized_shape) != [norm_dim]:
            continue

        weight_node = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        if weight_node is None or not isinstance(weight_node, torch.fx.Node):
            continue

        bias_node = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
        has_norm_bias = bias_node is not None and isinstance(bias_node, torch.fx.Node)

        eps = node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5)

        dtype = node.meta.get("dtype", torch.float32)

        has_add = (
            input_node.op == "call_function"
            and _aten_op_name(input_node.target) == "add"
        )

        if has_add:
            kernel_name = f"add_layer_norm_{counter}"
        else:
            kernel_name = f"layer_norm_{counter}"
        counter += 1

        kernel = AddLayerNormKernel(
            name=kernel_name,
            shape=list(shape),
            norm_dim=norm_dim,
            eps=eps,
            dtype=dtype,
            has_add=has_add,
            has_norm_bias=has_norm_bias,
        )

        if has_add:
            add_node = input_node
            a_node, b_node = add_node.args[0], add_node.args[1]

            graph.inserting_before(node)
            args = (a_node, b_node, weight_node)
            if has_norm_bias:
                args = args + (bias_node,)
            fused_node = graph.call_function(kernel, args)
            fused_node.name = _create_name(graph, kernel_name)
            _set_phase(fused_node, node.meta.get("phase", "forward"))
            fused_node.meta["shape"] = list(shape)
            fused_node.meta["dtype"] = dtype

            add_getitem = graph.call_function(operator.getitem, (fused_node, 0))
            add_getitem.name = _create_name(graph, f"{kernel_name}_add")
            add_getitem.meta["shape"] = list(shape)
            add_getitem.meta["dtype"] = dtype

            norm_getitem = graph.call_function(operator.getitem, (fused_node, 1))
            norm_getitem.name = _create_name(graph, f"{kernel_name}_norm")
            norm_getitem.meta["shape"] = list(shape)
            norm_getitem.meta["dtype"] = dtype

            node.replace_all_uses_with(norm_getitem)
            add_node.replace_all_uses_with(add_getitem)
            add_getitem.replace_input_with(add_getitem, fused_node)

            if not node.users:
                graph.erase_node(node)
            if not add_node.users:
                graph.erase_node(add_node)
        else:
            graph.inserting_before(node)
            args = (input_node, weight_node)
            if has_norm_bias:
                args = args + (bias_node,)
            fused_node = graph.call_function(kernel, args)
            fused_node.name = _create_name(graph, kernel_name)
            _set_phase(fused_node, node.meta.get("phase", "forward"))
            fused_node.meta["shape"] = list(shape)
            fused_node.meta["dtype"] = dtype

            node.replace_all_uses_with(fused_node)
            if not node.users:
                graph.erase_node(node)

        changed = True

    if changed:
        graph.lint()
        gm.recompile()
    return PassResult(gm=gm, changed=changed)

