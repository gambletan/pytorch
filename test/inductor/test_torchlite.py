"""
Torchlite convergence tests: reuse CommonTemplate from test_torchinductor.

Each test traces the function with torchlite, runs the traced graph, and compares
against eager execution. Tests that inspect inductor internals (generated code,
kernel counts, config patches, etc.) are auto-detected and skipped. Tests that
fail because torchlite's tracer doesn't handle every op yet are listed in
_torchlite_xfails.
"""

import copy
import inspect
import os
import sys

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import (
    clone_preserve_strides_offset,
    GPU_TYPE,
    HAS_GPU,
)
from torch._torchlite import compile, trace

# Adjust sys.path to find test_torchinductor in the same directory
test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_torchinductor import CommonTemplate, copy_tests, TestFailure


def _clone_inputs(inputs):
    return [
        x.clone().detach() if isinstance(x, torch.Tensor) else copy.deepcopy(x)
        for x in inputs
    ]


def check_torchlite(
    self,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    assert_equal=True,
    # Accepted but ignored: these are inductor-specific parameters that
    # CommonTemplate tests may pass through self.common(). Torchlite's
    # trace-and-replay pipeline doesn't need them.
    **_ignored,
):
    kwargs = kwargs or {}

    ref_inputs = _clone_inputs(example_inputs)
    trace_inputs = _clone_inputs(example_inputs)
    replay_inputs = _clone_inputs(example_inputs)

    torch.manual_seed(0)
    expected = model(*ref_inputs, **kwargs)

    gm = trace(model, trace_inputs)

    torch.manual_seed(0)
    actual = gm(*replay_inputs, **kwargs)

    if atol is None:
        atol = 1e-4
    if rtol is None:
        rtol = 1e-4

    if assert_equal:
        self.assertEqual(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
        self.assertEqual(
            replay_inputs, ref_inputs, atol=atol, rtol=rtol, equal_nan=True,
            msg="Input mutation mismatch between eager and torchlite",
        )


def check_torchlite_gpu(
    self,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    assert_equal=True,
    check_lowp=True,
    copy_to_gpu=True,
    **_ignored,
):
    if hasattr(model, "to"):
        model = copy.deepcopy(model)
        model = model.to(device=GPU_TYPE)

    if copy_to_gpu:
        example_inputs = [
            clone_preserve_strides_offset(x, device=GPU_TYPE)
            if isinstance(x, torch.Tensor)
            else x
            for x in example_inputs
        ]

    check_torchlite(
        self, model, example_inputs, kwargs,
        atol=atol, rtol=rtol, assert_equal=assert_equal,
    )

    if check_lowp:
        lowp_inputs = [
            x.to(torch.half)
            if isinstance(x, torch.Tensor) and x.dtype == torch.float32
            else x
            for x in example_inputs
        ]
        if hasattr(model, "to"):
            lowp_model = copy.deepcopy(model).to(torch.half)
        else:
            lowp_model = model

        lowp_atol = 1e-2 if atol is None else max(atol, 1e-2)
        lowp_rtol = 1e-2 if rtol is None else max(rtol, 1e-2)

        check_torchlite(
            self, lowp_model, lowp_inputs, kwargs,
            atol=lowp_atol, rtol=lowp_rtol, assert_equal=assert_equal,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect tests that use inductor internals or bypass self.common().
#
# Rather than maintaining a manual skip list of 280+ entries that drifts out of
# sync as CommonTemplate evolves (especially with parametrized tests), we
# inspect each test's source code for markers that indicate it cannot run under
# torchlite's trace-and-replay pipeline.
# ──────────────────────────────────────────────────────────────────────────────

# Markers that always indicate inductor-specific tests, regardless of whether
# self.common() is used.  These inspect generated code, check kernel counts,
# or use inductor-internal APIs.
_HARD_MARKERS = [
    "run_and_get_code",
    "run_and_get_triton_code",
    "run_and_get_cpp_code",
    "run_fw_bw_and_get_code",
    "assertGeneratedKernelCountEqual",
    "generated_kernel_count",
    "compile_fx_inner",
    "compile_fx",
    "FileCheck",
    "aoti_compile",
    "aoti_load",
    "register_ops_with_aoti_compile",
    "_run_and_assert_no_indirect_indexing",
    "metrics.ir_nodes",
    "metrics.generated",
    "DataTypePropagation",
    "_is_triggering_buffer_reuse",
    "_run_and_get_stripped_kernels",
    "get_post_grad_graph",
    "UniformValueConstantFolder",
    "triton_config_reduction",
]

# Markers that only matter when self.common() is NOT used.  A test that
# @config.patches an inductor setting but still routes through self.common()
# is fine — the config patches are irrelevant when self.common = check_torchlite.
_SOFT_MARKERS = [
    "config.patch",
    "inductor_config",
    "@config.",
]

_BYPASS_MARKERS = [
    "torch.compile(",
    "@torch.compile",
    "torch._dynamo.optimize",
    "optimize_assert",
]


def _is_inductor_specific(method):
    """Return True if this test method uses inductor internals or bypasses
    self.common() with direct torch.compile calls."""
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        return True

    if any(marker in src for marker in _HARD_MARKERS):
        return True

    uses_common = "self.common(" in src

    # Soft markers only matter when the test does NOT go through self.common()
    if not uses_common and any(marker in src for marker in _SOFT_MARKERS):
        return True

    # Direct torch.compile calls that bypass self.common()
    if not uses_common and any(marker in src for marker in _BYPASS_MARKERS):
        return True

    return False


_ALL_DEVICES = ("cpu", GPU_TYPE) if HAS_GPU else ("cpu",)


def _build_inductor_skip():
    skip = {}
    for name in CommonTemplate.__dict__:
        if not name.startswith("test_"):
            continue
        method = getattr(CommonTemplate, name)
        if _is_inductor_specific(method):
            skip[name] = TestFailure(_ALL_DEVICES, is_skip=True)
    return skip


_inductor_skip = _build_inductor_skip()


# ──────────────────────────────────────────────────────────────────────────────
# Expected failures: pure correctness tests that fail because torchlite's
# tracer doesn't handle every op yet. As torchlite gains coverage, tests
# should be removed from this list.
# ──────────────────────────────────────────────────────────────────────────────
_torchlite_xfails = {
    "test_cat_unbacked_legacy_empty": TestFailure(_ALL_DEVICES),
    "test_isin_tensor_scalar": TestFailure(_ALL_DEVICES),
    "test_mutable_custom_op_fixed_layout2": TestFailure(_ALL_DEVICES),
    "test_randn_generator": TestFailure(_ALL_DEVICES),
    "test_split_cumprod_low_prec": TestFailure(("cpu",)),
    "test_split_cumsum_low_prec": TestFailure(("cpu",)),
    "test_tmp_not_defined_issue3": TestFailure(_ALL_DEVICES),
}

_torchlite_gpu_xfails = {
    "test_angle": TestFailure((GPU_TYPE,)),
    "test_nll_loss_backward": TestFailure((GPU_TYPE,)),
    "test_pointwise_log_ndtr": TestFailure((GPU_TYPE,)),
    "test_softmax_backward_data": TestFailure((GPU_TYPE,)),
    "test_triton_kernel_bool_param": TestFailure((GPU_TYPE,)),
}

test_failures = {**_inductor_skip, **_torchlite_xfails, **_torchlite_gpu_xfails}


class TorchliteCpuTests(TestCase):
    common = check_torchlite
    device = "cpu"


copy_tests(CommonTemplate, TorchliteCpuTests, "cpu", test_failures)


if HAS_GPU:

    class TorchliteGpuTests(TestCase):
        common = check_torchlite_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, TorchliteGpuTests, GPU_TYPE, test_failures)


# ──────────────────────────────────────────────────────────────────────────────
# Compiled tests: full compile() pipeline (trace + run_passes + codegen).
#
# These test the same CommonTemplate patterns but through the complete
# compilation pipeline, catching bugs in passes and codegen that trace-only
# tests miss. Expected to have more xfails than trace-only tests since passes
# may fail on patterns the tracer handles fine.
# ──────────────────────────────────────────────────────────────────────────────


def check_torchlite_compiled(
    self,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    assert_equal=True,
    **_ignored,
):
    kwargs = kwargs or {}

    ref_inputs = _clone_inputs(example_inputs)
    compile_inputs = _clone_inputs(example_inputs)
    replay_inputs = _clone_inputs(example_inputs)

    torch.manual_seed(0)
    expected = model(*ref_inputs, **kwargs)

    compiled = compile(model, compile_inputs)

    torch.manual_seed(0)
    actual = compiled(*replay_inputs, **kwargs)

    if atol is None:
        atol = 1e-4
    if rtol is None:
        rtol = 1e-4

    if assert_equal:
        self.assertEqual(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
        self.assertEqual(
            replay_inputs, ref_inputs, atol=atol, rtol=rtol, equal_nan=True,
            msg="Input mutation mismatch between eager and torchlite compiled",
        )


def check_torchlite_compiled_gpu(
    self,
    model,
    example_inputs,
    kwargs=None,
    *,
    atol=None,
    rtol=None,
    assert_equal=True,
    check_lowp=True,
    copy_to_gpu=True,
    **_ignored,
):
    if hasattr(model, "to"):
        model = copy.deepcopy(model)
        model = model.to(device=GPU_TYPE)

    if copy_to_gpu:
        example_inputs = [
            clone_preserve_strides_offset(x, device=GPU_TYPE)
            if isinstance(x, torch.Tensor)
            else x
            for x in example_inputs
        ]

    check_torchlite_compiled(
        self, model, example_inputs, kwargs,
        atol=atol, rtol=rtol, assert_equal=assert_equal,
    )

    if check_lowp:
        lowp_inputs = [
            x.to(torch.half)
            if isinstance(x, torch.Tensor) and x.dtype == torch.float32
            else x
            for x in example_inputs
        ]
        if hasattr(model, "to"):
            lowp_model = copy.deepcopy(model).to(torch.half)
        else:
            lowp_model = model

        lowp_atol = 1e-2 if atol is None else max(atol, 1e-2)
        lowp_rtol = 1e-2 if rtol is None else max(rtol, 1e-2)

        check_torchlite_compiled(
            self, lowp_model, lowp_inputs, kwargs,
            atol=lowp_atol, rtol=lowp_rtol, assert_equal=assert_equal,
        )


# The compiled pipeline runs more passes than trace-only, so more tests
# may fail. Start with the trace-only xfails and add compiled-specific ones
# (mostly dynamize pass failures on multi-placeholder graphs and ops that
# don't decompose cleanly).
_torchlite_compiled_xfails = {
    **_torchlite_xfails,
    "test__dyn_quant_matmul_4bit_bf16_input": TestFailure(_ALL_DEVICES),
    "test__dyn_quant_matmul_4bit_fp32_input": TestFailure(_ALL_DEVICES),
    "test__unsafe_masked_index_put_accumulate": TestFailure(_ALL_DEVICES),
    "test_batch_norm_2d": TestFailure(_ALL_DEVICES),
    "test_bernoulli1_combo_kernels_False": TestFailure(_ALL_DEVICES),
    "test_bernoulli1_combo_kernels_True": TestFailure(_ALL_DEVICES),
    "test_complex_conv2d_conj": TestFailure(_ALL_DEVICES),
    "test_conv1d_depthwise": TestFailure(_ALL_DEVICES),
    "test_conv1d_with_permute": TestFailure(_ALL_DEVICES),
    "test_conv2d_channels_last": TestFailure(_ALL_DEVICES),
    "test_conv3d_channels_last_use_block_ptr_False": TestFailure(_ALL_DEVICES),
    "test_conv3d": TestFailure(_ALL_DEVICES),
    "test_conv_functional_bn_fuse": TestFailure(_ALL_DEVICES),
    "test_conv_with_as_strided": TestFailure(_ALL_DEVICES),
    "test_convolution1": TestFailure(_ALL_DEVICES),
    "test_convolution3": TestFailure(_ALL_DEVICES),
    "test_custom_op_fixed_layout_sequential": TestFailure(_ALL_DEVICES),
    "test_embedding": TestFailure(_ALL_DEVICES),
    "test_index_put3": TestFailure(_ALL_DEVICES),
    "test_index_put_fallback2": TestFailure(_ALL_DEVICES),
    "test_linear1": TestFailure(_ALL_DEVICES),
    "test_linear2": TestFailure(_ALL_DEVICES),
    "test_linear_float64": TestFailure(_ALL_DEVICES),
    "test_long_tensor": TestFailure(_ALL_DEVICES),
    "test_matmul_layer_norm": TestFailure(_ALL_DEVICES),
    "test_unsqueeze_inplace": TestFailure(_ALL_DEVICES),
    "test_upsample_cat_conv": TestFailure(_ALL_DEVICES),
    "test_view_on_aliased": TestFailure(_ALL_DEVICES),
}

_torchlite_compiled_gpu_xfails = {
    **_torchlite_gpu_xfails,
}

compiled_test_failures = {
    **_inductor_skip,
    **_torchlite_compiled_xfails,
    **_torchlite_compiled_gpu_xfails,
}


class TorchliteCompiledCpuTests(TestCase):
    common = check_torchlite_compiled
    device = "cpu"


copy_tests(CommonTemplate, TorchliteCompiledCpuTests, "cpu", compiled_test_failures)


if HAS_GPU:

    class TorchliteCompiledGpuTests(TestCase):
        common = check_torchlite_compiled_gpu
        device = GPU_TYPE

    copy_tests(
        CommonTemplate, TorchliteCompiledGpuTests, GPU_TYPE, compiled_test_failures
    )


class TestTorchliteVsTorchCompile(TestCase):
    """Sanity-check that torchlite.trace() and torch.compile(fullgraph=True)
    produce the same outputs for representative patterns."""

    def _compare(self, model, example_inputs, atol=1e-4, rtol=1e-4):
        ref_inputs = _clone_inputs(example_inputs)
        tl_inputs = _clone_inputs(example_inputs)
        tc_inputs = _clone_inputs(example_inputs)

        torch.manual_seed(0)
        gm = trace(model, _clone_inputs(example_inputs))
        tl_out = gm(*tl_inputs)

        torch.manual_seed(0)
        compiled_fn = torch.compile(model, fullgraph=True)
        tc_out = compiled_fn(*tc_inputs)

        self.assertEqual(tl_out, tc_out, atol=atol, rtol=rtol, equal_nan=True)

    def test_pointwise(self):
        def f(x):
            return torch.sin(x) * torch.cos(x) + x

        self._compare(f, [torch.randn(8, 16)])

    def test_reduction(self):
        def f(x):
            return x.sum(dim=-1)

        self._compare(f, [torch.randn(4, 8)])

    def test_matmul(self):
        def f(x, y):
            return x @ y

        self._compare(f, [torch.randn(4, 8), torch.randn(8, 16)])

    def test_mlp(self):
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(16, 32) * 0.01)
                self.w2 = nn.Parameter(torch.randn(32, 8) * 0.01)

            def forward(self, x):
                return torch.relu(x @ self.w1) @ self.w2

        torch.manual_seed(42)
        self._compare(MLP(), [torch.randn(4, 16)])

    def test_matmul_chain(self):
        def f(x, w1, w2):
            return (x @ w1) @ w2

        self._compare(f, [
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.randn(16, 4),
        ])

    def test_softmax(self):
        def f(x):
            return torch.softmax(x, dim=-1)

        self._compare(f, [torch.randn(4, 8)])

    def test_layer_norm_manual(self):
        def f(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            return (x - mean) / (var + 1e-5).sqrt()

        self._compare(f, [torch.randn(4, 16)])


class TestTorchliteInferencePipeline(TestCase):
    """End-to-end tests for the inference_codegen pipeline.

    These tests run models multiple times to check that buffer reuse via
    memory pools does not corrupt outputs on run 2+. They specifically
    cover the broadcast-fused-kernel bug where a RMSNorm-style kernel
    previously read past the end of rsqrt[M,1] and weight[N] inputs.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def _make_llama_ffn(self, d=64, h=128, seed=42):
        import torch.nn.functional as F

        class LlamaFFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.RMSNorm(d)
                self.gate_proj = torch.nn.Linear(d, h, bias=False)
                self.up_proj = torch.nn.Linear(d, h, bias=False)
                self.down_proj = torch.nn.Linear(h, d, bias=False)

            def forward(self, x):
                h_out = self.norm(x)
                return x + self.down_proj(
                    F.silu(self.gate_proj(h_out)) * self.up_proj(h_out)
                )

        torch.manual_seed(seed)
        return LlamaFFN().cuda().eval()

    @torch.no_grad()
    def test_llama_ffn_inference_pipeline_matches_eager(self):
        from torch._torchlite.api import inference_passes, run_passes, codegen
        from torch._torchlite.passes.triton import _TritonMatmulModule

        model = self._make_llama_ffn()
        x = torch.randn(8, 64, device="cuda")

        gm = trace(model, [x])
        pipeline = inference_passes(gm, [x])
        _TritonMatmulModule._backend_cache.clear()
        gm = run_passes(gm, [x], pipeline=pipeline)
        fn_tl = codegen(gm, inference_codegen=True, example_inputs=[x])

        ref = model(x)
        out1 = fn_tl(x)
        out2 = fn_tl(x)
        out3 = fn_tl(x)

        self.assertFalse(torch.isnan(out1).any(), "run 1 has NaN")
        self.assertFalse(torch.isnan(out2).any(), "run 2 has NaN")
        self.assertFalse(torch.isnan(out3).any(), "run 3 has NaN")
        self.assertEqual(out1, ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(out2, ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(out3, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
