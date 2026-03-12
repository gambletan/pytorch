"""Custom callable ops that appear as nodes in the FX graph.

These are non-torch operations used by torchlite passes:
save_for_backward (activation checkpointing), RNG state management
(save_rng_state, load_rng_state), and optimizer steps (adamw_step,
param_update).
"""
import torch


def _named(name):
    def decorator(fn):
        fn.__name__ = name
        fn.__qualname__ = name
        return fn
    return decorator


def param_update(param, new_value):
    param.data.copy_(new_value)


@_named("adamw_step")
def adamw_step(param, grad, m, v, step_t, lr, beta1, beta2, eps, weight_decay):
    # AdamW: decoupled weight decay + bias-corrected moment update.
    # step_t stays as a tensor (no .item()) so the entire computation
    # remains on-device and is capturable by CUDA graphs.
    if weight_decay != 0.0:
        param.data.mul_(1.0 - lr * weight_decay)
    m.data.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.data.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bc1 = torch.clamp(1.0 - beta1 ** step_t, min=1e-8)
    bc2 = torch.clamp(1.0 - beta2 ** step_t, min=1e-8)
    m_hat = m / bc1
    v_hat = v / bc2
    param.data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


@_named("save_rng_state")
def _save_rng_state():
    return torch.random.get_rng_state()


@_named("load_rng_state")
def _load_rng_state(state):
    torch.random.set_rng_state(state)


@_named("save_for_backward")
def _save_for_backward(x):
    return x


# FX codegen resolves call_function targets by __name__ / __qualname__
# within the module, so we need public aliases matching those names.
save_rng_state = _save_rng_state
load_rng_state = _load_rng_state
save_for_backward = _save_for_backward

_UNARY_POINTWISE_OPS = frozenset({
    "sin", "cos", "neg", "abs",
    "relu", "sigmoid", "tanh",
    "rsqrt", "sqrt", "exp", "log", "reciprocal",
    "silu", "gelu",
})

