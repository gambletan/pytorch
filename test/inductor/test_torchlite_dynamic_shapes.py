"""
Torchlite dynamic shapes tests: reuse CommonTemplate with doubled batch dim.

Each test traces at the original input shape, runs verify_graph + functionalize
+ dynamize passes, then replays at a different batch size and compares against
eager execution at that new shape. Tests that use inductor internals are
auto-skipped. Tests that fail due to shape-dependent logic or torchlite
limitations are listed in xfails.
"""

import copy
import os
import sys

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import (
    clone_preserve_strides_offset,
    GPU_TYPE,
    HAS_GPU,
)
from torch._torchlite import trace
from torch._torchlite.passes import dynamize, functionalize, verify_graph

test_dir = os.path.dirname(os.path.abspath(__file__))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

from test_torchinductor import CommonTemplate, copy_tests, TestFailure
from test_torchlite import _is_inductor_specific


def _rebatch(inputs, factor=2):
    """Create new inputs with batch dim scaled by factor."""
    out = []
    for x in inputs:
        if not isinstance(x, torch.Tensor) or x.ndim == 0:
            out.append(copy.deepcopy(x) if not isinstance(x, torch.Tensor) else x.clone())
            continue
        repeats = [1] * x.ndim
        repeats[0] = factor
        out.append(x.repeat(*repeats))
    return out


def check_torchlite_dynamic(
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

    def _clone_inputs(inputs):
        return [
            x.clone().detach() if isinstance(x, torch.Tensor) else copy.deepcopy(x)
            for x in inputs
        ]

    trace_inputs = _clone_inputs(example_inputs)
    gm = trace(model, trace_inputs)
    gm = verify_graph(gm, example_inputs).gm
    gm = functionalize(gm, example_inputs).gm
    gm = dynamize(gm, example_inputs).gm

    new_inputs = _rebatch(example_inputs)

    torch.manual_seed(0)
    expected = model(*_clone_inputs(new_inputs), **kwargs)

    torch.manual_seed(0)
    actual = gm(*_clone_inputs(new_inputs), **kwargs)

    if atol is None:
        atol = 1e-4
    if rtol is None:
        rtol = 1e-4

    if assert_equal:
        self.assertEqual(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


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

_torchlite_dynamic_xfails = {
    # Inherited from trace-only xfails (also fail with dynamic shapes)
    "test_cat_unbacked_legacy_empty": TestFailure(_ALL_DEVICES),
    "test_isin_tensor_scalar": TestFailure(_ALL_DEVICES),
    "test_mutable_custom_op_fixed_layout2": TestFailure(_ALL_DEVICES),
    "test_randn_generator": TestFailure(_ALL_DEVICES),
    "test_split_cumprod_low_prec": TestFailure(("cpu",)),
    "test_split_cumsum_low_prec": TestFailure(("cpu",)),
    "test_tmp_not_defined_issue3": TestFailure(_ALL_DEVICES),
    # _rebatch changes shape in ways that break shape-dependent ops
    # (square matrix assumptions, view-as-dtype byte counts, hardcoded
    # index tensors, convolution kernel/stride constraints, etc.)
    "test__unsafe_masked_index": TestFailure(_ALL_DEVICES),
    "test__unsafe_masked_index_put_accumulate": TestFailure(_ALL_DEVICES),
    "test_addmm": TestFailure(_ALL_DEVICES),
    "test_aliased_buffer_reuse": TestFailure(_ALL_DEVICES),
    "test_arange1": TestFailure(_ALL_DEVICES),
    "test_arange3": TestFailure(_ALL_DEVICES),
    "test_arange4": TestFailure(_ALL_DEVICES),
    "test_as_strided_scatter": TestFailure(_ALL_DEVICES),
    "test_bernoulli1_combo_kernels_False": TestFailure(_ALL_DEVICES),
    "test_bernoulli1_combo_kernels_True": TestFailure(_ALL_DEVICES),
    "test_bernoulli2": TestFailure(_ALL_DEVICES),
    "test_bitwise3": TestFailure(_ALL_DEVICES),
    "test_cat_extern_kernel": TestFailure(_ALL_DEVICES),
    "test_cat_uint8": TestFailure(_ALL_DEVICES),
    "test_cat_unbacked_2d": TestFailure(_ALL_DEVICES),
    "test_cat_unbacked_empty_1d": TestFailure(_ALL_DEVICES),
    "test_clamp_type_promotion": TestFailure(_ALL_DEVICES),
    "test_complex_conv2d_conj": TestFailure(_ALL_DEVICES),
    "test_constant_pad_fill_dtype": TestFailure(_ALL_DEVICES),
    "test_conv2d_backward_channels_last": TestFailure(_ALL_DEVICES),
    "test_convolution2": TestFailure(_ALL_DEVICES),
    "test_convolution4": TestFailure(_ALL_DEVICES),
    "test_convolution5": TestFailure(_ALL_DEVICES),
    "test_cumsum_pattern_matcher_issue": TestFailure(_ALL_DEVICES),
    "test_div_precision": TestFailure(_ALL_DEVICES),
    "test_embedding_bag": TestFailure(_ALL_DEVICES),
    "test_exact_stride": TestFailure(_ALL_DEVICES),
    "test_expand": TestFailure(_ALL_DEVICES),
    "test_flip_cat": TestFailure(_ALL_DEVICES),
    "test_float_index_expression_type_promotion": TestFailure(_ALL_DEVICES),
    "test_fuse_tiled": TestFailure(_ALL_DEVICES),
    "test_gather1": TestFailure(_ALL_DEVICES),
    "test_horizonal_fusion1": TestFailure(_ALL_DEVICES),
    "test_index1": TestFailure(_ALL_DEVICES),
    "test_index_propagation_device_assert_masked": TestFailure(_ALL_DEVICES),
    "test_index_put3": TestFailure(_ALL_DEVICES),
    "test_index_put_fallback2": TestFailure(_ALL_DEVICES),
    "test_index_put_index": TestFailure(_ALL_DEVICES),
    "test_inplace_where_pointwise": TestFailure(_ALL_DEVICES),
    "test_int8_weight_only_quant": TestFailure(_ALL_DEVICES),
    "test_isinf2": TestFailure(_ALL_DEVICES),
    "test_lerp": TestFailure(_ALL_DEVICES),
    "test_linalg_eig_stride_consistency": TestFailure(_ALL_DEVICES),
    "test_masked_fill": TestFailure(_ALL_DEVICES),
    "test_matmul_layer_norm": TestFailure(_ALL_DEVICES),
    "test_mixed_mm": TestFailure(_ALL_DEVICES),
    "test_mixed_mm2": TestFailure(_ALL_DEVICES),
    "test_mixed_mm3": TestFailure(_ALL_DEVICES),
    "test_neg_max_uint8": TestFailure(_ALL_DEVICES),
    "test_nonzero_unbacked_refinement": TestFailure(_ALL_DEVICES),
    "test_pad_view": TestFailure(_ALL_DEVICES),
    "test_pow_by_natural_log2_dynamic_shapes": TestFailure(_ALL_DEVICES),
    "test_remove_noop_copy": TestFailure(_ALL_DEVICES),
    "test_repeat_interleave_2": TestFailure(_ALL_DEVICES),
    "test_sdpa_unaligned_mask": TestFailure(_ALL_DEVICES),
    "test_select_scatter": TestFailure(_ALL_DEVICES),
    "test_shape_padding": TestFailure(_ALL_DEVICES),
    "test_shape_prop_torch_ones": TestFailure(_ALL_DEVICES),
    "test_slice4": TestFailure(_ALL_DEVICES),
    "test_slice_scatter5": TestFailure(_ALL_DEVICES),
    "test_stack": TestFailure(_ALL_DEVICES),
    "test_sum3": TestFailure(_ALL_DEVICES),
    "test_sum_dtype": TestFailure(_ALL_DEVICES),
    "test_tensor2": TestFailure(_ALL_DEVICES),
    "test_tensor3": TestFailure(_ALL_DEVICES),
    "test_to_device_constant": TestFailure(_ALL_DEVICES),
    "test_transpose": TestFailure(_ALL_DEVICES),
    "test_uint4x2_mixed_mm": TestFailure(_ALL_DEVICES),
    "test_unbacked_floordiv_simplify": TestFailure(_ALL_DEVICES),
    "test_unbind": TestFailure(_ALL_DEVICES),
    "test_unsqueeze_inplace": TestFailure(_ALL_DEVICES),
    "test_upsample_nearest2d_backward": TestFailure(_ALL_DEVICES),
    "test_views1": TestFailure(_ALL_DEVICES),
    "test_views3": TestFailure(_ALL_DEVICES),
    "test_views4": TestFailure(_ALL_DEVICES),
    "test_views5": TestFailure(_ALL_DEVICES),
    "test_zeros": TestFailure(_ALL_DEVICES),
}

# view-as-dtype tests fail when _rebatch changes the batch dim in ways
# that break dtype reinterpretation views (element count must be divisible
# by the new element size). Tests where src has >= bytes-per-element as dst
# tend to pass because doubling the batch still produces valid element counts.
_DTYPEVIEW_TYPES = [
    "bfloat16", "float16", "float32", "float64",
    "int8", "int16", "int32", "int64", "uint8",
]
_DTYPEVIEW_PASS = {
    "test_dtypeview_float32_int8",
    "test_dtypeview_float32_uint8",
    "test_dtypeview_float64_bfloat16",
    "test_dtypeview_float64_float16",
    "test_dtypeview_float64_int16",
    "test_dtypeview_float64_int8",
    "test_dtypeview_float64_uint8",
    "test_dtypeview_int32_int8",
    "test_dtypeview_int32_uint8",
    "test_dtypeview_int64_bfloat16",
    "test_dtypeview_int64_float16",
    "test_dtypeview_int64_int16",
    "test_dtypeview_int64_int8",
    "test_dtypeview_int64_uint8",
}
for _src in _DTYPEVIEW_TYPES:
    for _dst in _DTYPEVIEW_TYPES:
        _name = f"test_dtypeview_{_src}_{_dst}"
        if _name not in _torchlite_dynamic_xfails and _name not in _DTYPEVIEW_PASS:
            _torchlite_dynamic_xfails[_name] = TestFailure(_ALL_DEVICES)

_torchlite_dynamic_gpu_xfails = {
    "test_angle": TestFailure((GPU_TYPE,)),
    "test_nll_loss_backward": TestFailure((GPU_TYPE,)),
    "test_pointwise_log_ndtr": TestFailure((GPU_TYPE,)),
    "test_softmax_backward_data": TestFailure((GPU_TYPE,)),
    "test_triton_kernel_bool_param": TestFailure((GPU_TYPE,)),
}

test_failures = {
    **_inductor_skip,
    **_torchlite_dynamic_xfails,
    **_torchlite_dynamic_gpu_xfails,
}


def check_torchlite_dynamic_gpu(
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

    check_torchlite_dynamic(
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

        check_torchlite_dynamic(
            self, lowp_model, lowp_inputs, kwargs,
            atol=lowp_atol, rtol=lowp_rtol, assert_equal=assert_equal,
        )


class TorchliteDynamicCpuTests(TestCase):
    common = check_torchlite_dynamic
    device = "cpu"


copy_tests(CommonTemplate, TorchliteDynamicCpuTests, "cpu", test_failures)


if HAS_GPU:

    class TorchliteDynamicGpuTests(TestCase):
        common = check_torchlite_dynamic_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, TorchliteDynamicGpuTests, GPU_TYPE, test_failures)

if __name__ == "__main__":
    run_tests()
