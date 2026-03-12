"""Torchlite: a from-scratch compiler for PyTorch models.

The compiler has three phases:
  trace()      - capture a model into an FX graph
  run_passes() - run all graph transformation passes
  codegen()    - convert the transformed graph into a callable
compile() = trace() + run_passes() + codegen().
"""
from torch._torchlite import passes  # noqa: F401
from torch._torchlite.passes import (
    activation_checkpoint,
    annotate_dtensor,
    autograd_per_op,
    cudagraph_partition,
    decompose,
    dynamize,
    fsdp_unwrap,
    functionalize,
    fuse,
    FusedKernel,
    FusedOp,
    FusionGroup,
    memory_plan,
    normalize,
    save_activations,
    optimizer,
    PassResult,
    precompile,
    rng_functionalize,
    subclass_unwrap,
    triton_codegen,
    verify_graph,
)

from torch._torchlite.api import (
    codegen,
    codegen_inference,
    compile,
    default_passes,
    inference_passes,
    precompile_load,
    precompile_save,
    run_passes,
    timed_run_passes,
    trace,
)

__all__ = [
    # Entry points (trace → run_passes → codegen)
    "trace",
    "run_passes",
    "timed_run_passes",
    "default_passes",
    "inference_passes",
    "codegen",
    "codegen_inference",
    "compile",
    "precompile_save",
    "precompile_load",
    # Passes submodule
    "passes",
    # Pass result type
    "PassResult",
    # Individual graph passes (canonical pipeline order)
    "cudagraph_partition",
    "normalize",
    "verify_graph",
    "functionalize",
    "dynamize",
    "annotate_dtensor",
    "subclass_unwrap",
    "fsdp_unwrap",
    "autograd_per_op",
    "rng_functionalize",
    "save_activations",
    "activation_checkpoint",
    "optimizer",
    "memory_plan",
    "decompose",
    "fuse",
    "triton_codegen",
    "precompile",
    # Data types
    "FusedKernel",
    "FusedOp",
    "FusionGroup",
]
