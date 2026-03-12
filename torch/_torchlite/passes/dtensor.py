"""DTensor-related passes: subclass_unwrap and fsdp_unwrap."""
import operator
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.collectives import (
    _allgather,
    _allreduce,
    _BINARY_ELEMENTWISE_OPS,
    _has_dtensor_params,
    _is_replicate,
    _is_shard,
    _MATMUL_OPS,
    _NORM_OPS,
    _POINTWISE_PLACEMENT_OPS,
    _propagate_binary,
    _propagate_embedding,
    _propagate_matmul,
    _propagate_norm,
    _propagate_pointwise,
    _propagate_reduction,
    _propagate_view,
    _REDUCTION_OPS,
    _reduce_scatter,
    _VIEW_OPS,
)
from torch._torchlite.passes.common import (
    _create_name,
    _deep_getattr,
    _graph_meta,
    _set_phase,
    PassResult,
)


def subclass_unwrap(gm, example_inputs, *, world_size=2):
    graph = gm.graph

    placement = {}
    for node in graph.nodes:
        if node.op not in ("get_attr", "placeholder"):
            continue
        spec = node.meta.get("dtensor_spec")
        if spec:
            placement[node] = spec

    if not placement:
        return PassResult(gm=gm)

    for node in graph.nodes:
        if node in placement or node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")

        if node.target is operator.getitem:
            parent = node.args[0] if node.args else None
            if isinstance(parent, torch.fx.Node) and parent in placement:
                placement[node] = placement[parent]
            continue

        if name in _POINTWISE_PLACEMENT_OPS:
            _propagate_pointwise(node, placement)
        elif name in _BINARY_ELEMENTWISE_OPS:
            _propagate_binary(node, placement)
        elif name in _MATMUL_OPS:
            _propagate_matmul(node, placement)
        elif name in _REDUCTION_OPS:
            _propagate_reduction(node, placement)
        elif name in _VIEW_OPS:
            _propagate_view(node, placement)
        elif name in _NORM_OPS:
            _propagate_norm(node, placement)
        elif name == "embedding":
            _propagate_embedding(node, placement)

    # Nodes that have been created by collective insertion and should not
    # have their shapes adjusted in the final shard-dim division pass.
    collective_nodes = set()

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        spec = placement.get(node)
        if spec is None or spec[0] != "_Partial":
            continue
        users = list(node.users)
        graph.inserting_after(node)
        ar = graph.call_function(_allreduce, (node,))
        ar.name = _create_name(graph, "allreduce")
        _set_phase(ar, node.meta.get("phase", "forward"))
        if "shape" in node.meta:
            ar.meta["shape"] = list(node.meta["shape"])
        for user in users:
            if user.op == "call_function" and user.target is torch.Tensor.copy_:
                continue
            user.replace_input_with(node, ar)
        placement[ar] = ("Replicate", None)
        collective_nodes.add(ar)

    _BINARY_ALLGATHER_OPS = frozenset({
        "sub", "add", "mul", "div", "pow",
        "where", "maximum", "minimum",
        "eq", "ne", "lt", "le", "gt", "ge",
    })

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if name not in _BINARY_ALLGATHER_OPS:
            continue

        tensor_args = [
            (a, placement.get(a))
            for a in node.args
            if isinstance(a, torch.fx.Node)
        ]
        if len(tensor_args) < 2:
            continue

        a_node, a_spec = tensor_args[0]
        b_node, b_spec = tensor_args[1]

        target_node = None
        target_spec = None
        if _is_shard(a_spec) and (_is_replicate(b_spec) or b_spec is None):
            target_node = a_node
            target_spec = a_spec
        elif _is_shard(b_spec) and (_is_replicate(a_spec) or a_spec is None):
            target_node = b_node
            target_spec = b_spec

        if target_node is not None:
            graph.inserting_before(node)
            ag = graph.call_function(_allgather, (target_node,))
            ag.name = _create_name(graph, "allgather")
            _set_phase(ag, node.meta.get("phase", "forward"))
            # Allgather output has the full (global) shape: restore the
            # shard dim to its pre-sharding size.
            if "shape" in target_node.meta:
                gathered_shape = list(target_node.meta["shape"])
                shard_dim = target_spec[1]
                gathered_shape[shard_dim] = gathered_shape[shard_dim] * world_size
                ag.meta["shape"] = gathered_shape
            node.replace_input_with(target_node, ag)
            placement[ag] = ("Replicate", None)
            placement[node] = ("Replicate", None)
            collective_nodes.add(ag)

    for node in graph.nodes:
        spec = placement.get(node)
        shape = node.meta.get("shape")
        if spec:
            node.meta["dtensor_spec"] = spec
        if node in collective_nodes:
            continue
        if spec and shape and _is_shard(spec):
            dim = spec[1]
            local_shape = list(shape)
            if local_shape[dim] % world_size != 0:
                raise ValueError(
                    f"subclass_unwrap: node '{node.name}' shard dim {dim} "
                    f"(size {local_shape[dim]}) is not evenly divisible by "
                    f"world_size={world_size}."
                )
            local_shape[dim] = local_shape[dim] // world_size
            node.meta["shape"] = local_shape

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)


def fsdp_unwrap(gm, example_inputs, *, dp_dim=0, world_size=2):
    """FSDP pass: insert allgather before forward uses of DP-sharded parameters
    and reduce_scatter after backward gradient computation.

    Parameters sharded on `dp_dim` (Shard(dp_dim)) are identified as FSDP
    parameters. The pass:
    1. Inserts allgather before forward parameter usage
    2. Inserts reduce_scatter after backward gradient computation
    3. Handles reshard_after_forward: frees full params after forward
    """
    graph = gm.graph

    fsdp_params = {}
    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        spec = node.meta.get("dtensor_spec")
        if spec is not None and spec[0] == "Shard" and spec[1] == dp_dim:
            fsdp_params[node] = spec

    if not fsdp_params:
        return PassResult(gm=gm)

    for param_node in list(fsdp_params):
        fwd_users = [
            u for u in param_node.users
            if u.op == "call_function"
            and u.meta.get("phase", "forward") == "forward"
        ]

        if not fwd_users:
            continue

        first_user = fwd_users[0]
        graph.inserting_before(first_user)
        ag = graph.call_function(_allgather, (param_node,))
        ag.name = _create_name(
            graph, f"fsdp_ag_{param_node.target.split('.')[-1]}"
        )
        _set_phase(ag, "forward")
        if "shape" in param_node.meta:
            shape = list(param_node.meta["shape"])
            shape[dp_dim] = shape[dp_dim] * world_size
            ag.meta["shape"] = shape
        ag.meta["dtensor_spec"] = ("Replicate", None)

        for user in fwd_users:
            user.replace_input_with(param_node, ag)

    param_grad_info = _graph_meta(gm.graph).get("param_grad_info", {})
    if param_grad_info:
        output_node = None
        for n in graph.nodes:
            if n.op == "output":
                output_node = n
                break

        orig_output = output_node.args[0]
        if isinstance(orig_output, (tuple, list)):
            param_targets_set = {n.target for n in fsdp_params}
            output_replacements = {}
            for param_name, grad_idx in param_grad_info.items():
                if param_name not in param_targets_set:
                    continue
                grad_node = orig_output[1 + grad_idx]
                if not isinstance(grad_node, torch.fx.Node):
                    continue

                users = list(grad_node.users)
                graph.inserting_after(grad_node)
                rs = graph.call_function(_reduce_scatter, (grad_node,))
                rs.name = _create_name(
                    graph, f"fsdp_rs_{param_name.split('.')[-1]}"
                )
                _set_phase(rs, "backward")
                if "shape" in grad_node.meta:
                    shape = list(grad_node.meta["shape"])
                    if shape[dp_dim] % world_size != 0:
                        raise ValueError(
                            f"fsdp_unwrap: grad for '{param_name}' dim {dp_dim} "
                            f"(size {shape[dp_dim]}) is not evenly divisible by "
                            f"world_size={world_size}."
                        )
                    shape[dp_dim] = shape[dp_dim] // world_size
                    rs.meta["shape"] = shape
                rs.meta["dtensor_spec"] = ("Shard", dp_dim)

                for user in users:
                    if user is not rs and user.op != "output":
                        user.replace_input_with(grad_node, rs)

                output_replacements[1 + grad_idx] = rs

            if output_replacements:
                new_output = list(orig_output)
                for idx, rs in output_replacements.items():
                    new_output[idx] = rs
                output_node.args = (tuple(new_output),)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
