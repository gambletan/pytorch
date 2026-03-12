"""Optimizer pass: insert SGD or AdamW weight updates into the graph."""
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _deep_getattr,
    _graph_meta,
    _set_phase,
    PassResult,
)
from torch._torchlite.ops import adamw_step, param_update


def _emit_sgd_update(graph, param_node, grad_node, short, lr):
    scaled_grad = graph.call_function(torch.mul, (lr, grad_node))
    scaled_grad.name = _create_name(graph, short + "_step")
    _set_phase(scaled_grad, "optimizer")

    new_param = graph.call_function(torch.sub, (param_node, scaled_grad))
    new_param.name = _create_name(graph, short + "_new")
    _set_phase(new_param, "optimizer")

    copy_node = graph.call_function(
        param_update, (param_node, new_param)
    )
    copy_node.name = _create_name(graph, short + "_update")
    _set_phase(copy_node, "optimizer")


def optimizer(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> PassResult:
    graph = gm.graph
    param_grad_info = _graph_meta(gm.graph).get("param_grad_info", {})
    if not param_grad_info:
        has_params = any(n.op == "get_attr" for n in graph.nodes)
        has_bwd = any(
            n.op == "call_function" and n.meta.get("phase") == "backward"
            for n in graph.nodes
        )
        if has_params and has_bwd:
            raise RuntimeError(
                "optimizer: the graph has parameters and backward nodes but "
                "param_grad_info is empty. This usually means the Graph "
                "object was garbage-collected and re-created between "
                "autograd_per_op and optimizer, causing the WeakKeyDictionary "
                "metadata to be lost. Ensure the GraphModule stays alive "
                "across the full pass pipeline."
            )
        return PassResult(gm=gm)

    output_node = None
    for n in graph.nodes:
        if n.op == "output":
            output_node = n
            break

    orig_output = output_node.args[0]
    if not isinstance(orig_output, (tuple, list)):
        return PassResult(gm=gm)

    param_nodes = {}
    for n in graph.nodes:
        if n.op == "get_attr":
            param_nodes[n.target] = n

    graph.inserting_before(output_node)

    if optimizer_type == "adamw":
        adam_state = {}
        step_tensor = torch.tensor(0, dtype=torch.long)
        step_target = "_adam_step"
        gm.register_buffer(step_target, step_tensor, persistent=True)
        step_node = graph.get_attr(step_target)
        _set_phase(step_node, "optimizer")

        for param_name in param_grad_info:
            param_val = _deep_getattr(gm, param_name)
            m = torch.zeros_like(param_val)
            v = torch.zeros_like(param_val)
            m_target = f"_adam_m_{param_name.replace('.', '_')}"
            v_target = f"_adam_v_{param_name.replace('.', '_')}"
            gm.register_buffer(m_target, m, persistent=True)
            gm.register_buffer(v_target, v, persistent=True)

            m_node = graph.get_attr(m_target)
            _set_phase(m_node, "optimizer")
            v_node = graph.get_attr(v_target)
            _set_phase(v_node, "optimizer")
            adam_state[param_name] = (m_node, v_node)

        step_inc = graph.call_function(torch.add, (step_node, 1))
        step_inc.name = _create_name(graph, "adam_step_inc")
        _set_phase(step_inc, "optimizer")
        step_copy = graph.call_function(torch.Tensor.copy_, (step_node, step_inc))
        step_copy.name = _create_name(graph, "adam_step_update")
        _set_phase(step_copy, "optimizer")

        for param_name, grad_idx in param_grad_info.items():
            param_node = param_nodes[param_name]
            grad_node = orig_output[1 + grad_idx]
            m_node, v_node = adam_state[param_name]
            short = param_name.split(".")[-1]

            update_node = graph.call_function(
                adamw_step,
                (
                    param_node, grad_node,
                    m_node, v_node, step_inc,
                    lr, betas[0], betas[1], eps, weight_decay,
                ),
            )
            update_node.name = _create_name(graph, short + "_adam_update")
            _set_phase(update_node, "optimizer")

    else:
        for param_name, grad_idx in param_grad_info.items():
            param_node = param_nodes[param_name]
            grad_node = orig_output[1 + grad_idx]
            short = param_name.split(".")[-1]
            _emit_sgd_update(graph, param_node, grad_node, short, lr)

    loss_node = orig_output[0]
    output_node.args = (loss_node,)

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
