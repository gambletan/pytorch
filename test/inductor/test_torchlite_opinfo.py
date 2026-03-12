"""
Torchlite OpInfo tests: run every op in op_db through torchlite.trace() and
compare against eager execution.

This tests torchlite's tracing coverage across ~500 ops with varied dtypes,
and gradient correctness for ops that support autograd.
"""

import copy
import os
from collections import defaultdict
from functools import partial

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db, skipOps
from torch._torchlite import trace

f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool

_ops = partial(
    ops,
    dtypes=OpDTypes.supported,
    allowed_dtypes=[f32, f64, i32, i64, b8],
)

_grad_ops = partial(
    ops,
    dtypes=OpDTypes.supported,
    allowed_dtypes=[f32, f64],
)

START = os.getenv("PYTORCH_TEST_RANGE_START", None)
END = os.getenv("PYTORCH_TEST_RANGE_END", None)
if START is not None or END is not None:
    assert END is not None
    assert START is not None
    START = int(START)
    END = int(END)
    assert START < END
else:
    START = 0
    END = len(op_db)

COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"


# ──────────────────────────────────────────────────────────────────────────────
# Skip / xfail dictionaries.
#
# To populate, run with PYTORCH_COLLECT_EXPECT=1 and paste the output.
# ──────────────────────────────────────────────────────────────────────────────
_all_dtypes = {b8, f32, f64, i32, i64}

torchlite_skips = defaultdict(dict)
_empty_skips = {
    "empty": _all_dtypes,
    "empty_like": _all_dtypes,
    "empty_permuted": _all_dtypes,
    "empty_strided": _all_dtypes,
    "new_empty": _all_dtypes,
    "new_empty_strided": _all_dtypes,
}
torchlite_skips["cpu"] = {**_empty_skips}
torchlite_skips["cuda"] = {**_empty_skips}

torchlite_expected_failures = defaultdict(dict)
torchlite_expected_failures["cpu"] = {
    # --- Tensor property descriptors ---
    "H": {b8, f32, f64, i32, i64},
    "T": {b8, f32, f64, i32, i64},
    # --- Multi-tensor input / reduction ops ---
    ("_segment_reduce", "lengths"): {f32, f64},
    ("_segment_reduce", "offsets"): {f32, f64},
    "_upsample_bilinear2d_aa": {f32, f64},
    "addbmm": {f32, f64, i32, i64},
    "addmm": {f32, f64, i32, i64},
    ("addmm", "decomposed"): {f32, f64, i32, i64},
    "addmv": {f32, f64, i32, i64},
    "allclose": {f32, f64},
    # --- Factory ops ---
    "arange": {f32, f64, i32, i64},
    # --- View / stride ops ---
    "as_strided": {b8, f32, f64, i32, i64},
    "as_strided_copy": {b8, f32, f64, i32, i64},
    "as_strided_scatter": {b8, f32, f64, i32, i64},
    # --- Multi-tensor input ops ---
    "baddbmm": {f32, f64, i32, i64},
    # --- RNG ops ---
    "bernoulli": {f32, f64},
    # --- Multi-tensor input ops ---
    "bincount": {i32, i64},
    "bucketize": {f32, f64, i32, i64},
    "cat": {b8, f32, f64, i32, i64},
    # --- RNG ops ---
    "cauchy": {f32, f64},
    # --- Linalg ops ---
    "cholesky": {f32, f64},
    # --- Multi-tensor input ops ---
    "combinations": {b8, f32, f64, i32, i64},
    # --- In-place / mutation ops ---
    ("div", "floor_rounding"): {f32, f64, i32, i64},
    ("div", "trunc_rounding"): {f32, f64, i32, i64},
    # --- RNG ops ---
    "exponential": {f32, f64},
    # --- Factory ops ---
    "eye": {b8, f32, f64, i32, i64},
    # --- FFT ops ---
    "fft.fft": {b8, f32, f64, i32, i64},
    "fft.fft2": {b8, f32, f64, i32, i64},
    "fft.fftn": {b8, f32, f64, i32, i64},
    "fft.hfft": {b8, f32, f64, i32, i64},
    "fft.hfft2": {b8, f32, f64, i32, i64},
    "fft.hfftn": {b8, f32, f64, i32, i64},
    "fft.ifft": {b8, f32, f64, i32, i64},
    "fft.ifft2": {b8, f32, f64, i32, i64},
    "fft.ifftn": {b8, f32, f64, i32, i64},
    "fft.ihfft": {b8, f32, f64, i32, i64},
    "fft.ihfft2": {b8, f32, f64, i32, i64},
    "fft.ihfftn": {b8, f32, f64, i32, i64},
    "fft.irfft": {b8, f32, f64, i32, i64},
    "fft.irfft2": {b8, f32, f64, i32, i64},
    "fft.irfftn": {b8, f32, f64, i32, i64},
    "fft.rfft": {b8, f32, f64, i32, i64},
    "fft.rfft2": {b8, f32, f64, i32, i64},
    "fft.rfftn": {b8, f32, f64, i32, i64},
    # --- In-place / mutation ops ---
    "fill": {b8, f32, f64, i32, i64},
    "flip": {b8, f32, f64, i32, i64},
    # --- Factory ops ---
    "full": {b8, f32, f64, i32, i64},
    # --- RNG ops ---
    "geometric": {f32, f64, i32, i64},
    # --- Multi-tensor input ops ---
    "gradient": {f32, f64, i32, i64},
    "histc": {f32, f64},
    "histogram": {f32, f64},
    "histogramdd": {f32, f64},
    "index_add": {b8, f32, f64, i32, i64},
    "index_put": {b8, f32, f64, i32, i64},
    ("index_reduce", "amax"): {f32, f64, i32, i64},
    ("index_reduce", "amin"): {f32, f64, i32, i64},
    ("index_reduce", "mean"): {f32, f64, i32, i64},
    ("index_reduce", "prod"): {f32, f64, i32, i64},
    # --- Linalg ops ---
    "linalg.cholesky": {f32, f64},
    "linalg.cholesky_ex": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.ldl_factor_ex": {f32, f64},
    "linalg.ldl_solve": {f32, f64},
    "linalg.lstsq": {f32, f64},
    ("linalg.lstsq", "grad_oriented"): {f32, f64},
    "linalg.lu": {f32, f64},
    "linalg.lu_factor": {f32, f64},
    "linalg.lu_factor_ex": {f32, f64},
    "linalg.lu_solve": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    ("linalg.matrix_rank", "hermitian"): {f32, f64},
    "linalg.norm": {f32, f64},
    ("linalg.norm", "subgradients_at_zero"): {f32, f64},
    "linalg.pinv": {f32, f64},
    ("linalg.pinv", "hermitian"): {f32, f64},
    "linalg.solve_triangular": {f32, f64},
    "linalg.svd": {f32, f64},
    "linalg.tensorinv": {f32, f64},
    "linalg.tensorsolve": {f32, f64},
    "linalg.vander": {f32, f64, i32, i64},
    "linalg.vector_norm": {f32, f64},
    # --- Factory ops ---
    "linspace": {f32, f64, i32, i64},
    ("linspace", "tensor_overload"): {f32, f64, i32, i64},
    # --- RNG ops ---
    "log_normal": {f32, f64},
    ("log_softmax", "with_dtype"): {b8, f32, f64, i32, i64},
    # --- Factory ops ---
    "logspace": {f32, f64, i32, i64},
    ("logspace", "tensor_overload"): {f32, f64, i32, i64},
    # --- Tensor property descriptors ---
    "mH": {b8, f32, f64, i32, i64},
    "mT": {b8, f32, f64, i32, i64},
    # --- Masked ops ---
    "masked.amax": {f32, f64, i32, i64},
    "masked.amin": {f32, f64, i32, i64},
    "masked.argmax": {f32, f64, i32, i64},
    "masked.argmin": {f32, f64, i32, i64},
    "masked.cumprod": {f32, f64, i32, i64},
    "masked.cumsum": {f32, f64, i32, i64},
    "masked.log_softmax": {f32, f64},
    "masked.logaddexp": {f32, f64},
    "masked.logsumexp": {f32, f64, i32, i64},
    "masked.mean": {f32, f64},
    "masked.median": {f32, f64},
    "masked.norm": {f32, f64},
    "masked.prod": {b8, f32, f64, i32, i64},
    "masked.softmax": {f32, f64},
    "masked.softmin": {f32, f64},
    "masked.std": {f32, f64, i32, i64},
    "masked.sum": {b8, f32, f64, i32, i64},
    "masked.var": {f32, f64, i32, i64},
    # --- Multi-tensor input ops ---
    "max_pool2d_with_indices_backward": {f32, f64},
    ("meshgrid", "list_of_tensors"): {b8, f32, f64, i32, i64},
    ("meshgrid", "variadic_tensors"): {b8, f32, f64, i32, i64},
    # --- RNG ops ---
    "multinomial": {f32, f64},
    "nn.functional.alpha_dropout": {f32, f64},
    # --- Multi-tensor input / NN ops ---
    "nn.functional.batch_norm": {f32, f64},
    "nn.functional.conv1d": {f32, f64, i64},
    "nn.functional.conv2d": {f32, f64, i64},
    "nn.functional.conv3d": {f32, f64, i64},
    "nn.functional.conv_transpose1d": {f32, f64, i64},
    "nn.functional.conv_transpose2d": {f32, f64, i64},
    "nn.functional.conv_transpose3d": {f32, f64, i64},
    "nn.functional.cosine_embedding_loss": {b8, f32, f64, i32, i64},
    "nn.functional.cosine_similarity": {f32, f64},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f32, f64},
    "nn.functional.dropout2d": {f32, f64},
    "nn.functional.dropout3d": {f32, f64},
    "nn.functional.embedding_bag": {f32, f64},
    ("nn.functional.feature_alpha_dropout", "with_train"): {f32, f64},
    ("nn.functional.feature_alpha_dropout", "without_train"): {b8, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f32, f64},
    "nn.functional.fractional_max_pool3d": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.gelu": {f32, f64},
    "nn.functional.grid_sample": {f32, f64},
    "nn.functional.group_norm": {f32, f64},
    "nn.functional.hardshrink": {f32, f64},
    "nn.functional.hardtanh": {f32, f64, i32, i64},
    "nn.functional.hinge_embedding_loss": {f32, f64},
    "nn.functional.huber_loss": {f32, f64},
    "nn.functional.instance_norm": {f32, f64},
    ("nn.functional.interpolate", "area"): {f32, f64},
    ("nn.functional.interpolate", "bicubic"): {f32, f64},
    ("nn.functional.interpolate", "bilinear"): {f32, f64},
    ("nn.functional.interpolate", "linear"): {f32, f64},
    ("nn.functional.interpolate", "nearest-exact"): {f32, f64},
    ("nn.functional.interpolate", "nearest"): {f32, f64},
    ("nn.functional.interpolate", "trilinear"): {f32, f64},
    "nn.functional.kl_div": {f32, f64},
    "nn.functional.layer_norm": {f32, f64},
    "nn.functional.local_response_norm": {f32, f64, i64},
    "nn.functional.margin_ranking_loss": {f32, f64, i32, i64},
    "nn.functional.max_pool1d": {f32, f64},
    "nn.functional.max_pool2d": {f32, f64, i32, i64},
    "nn.functional.max_pool3d": {f32, f64, i32, i64},
    "nn.functional.max_unpool1d": {f32, f64},
    ("nn.functional.max_unpool1d", "grad"): {f32, f64},
    "nn.functional.max_unpool2d": {f32, f64},
    ("nn.functional.max_unpool2d", "grad"): {f32, f64},
    "nn.functional.max_unpool3d": {f32, f64},
    ("nn.functional.max_unpool3d", "grad"): {f32, f64},
    "nn.functional.multi_head_attention_forward": {f32, f64},
    "nn.functional.nll_loss": {f32, f64},
    "nn.functional.normalize": {f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pixel_shuffle": {b8, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f32, f64, i32, i64},
    "nn.functional.poisson_nll_loss": {f32, f64, i32, i64},
    "nn.functional.prelu": {f32, f64},
    "nn.functional.rms_norm": {f32, f64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.scaled_dot_product_attention": {f32, f64},
    ("nn.functional.softmin", "with_dtype"): {f32, f64, i32, i64},
    "nn.functional.softplus": {f32, f64},
    "nn.functional.softshrink": {f32, f64},
    "nn.functional.triplet_margin_loss": {f32, f64, i32, i64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64},
    "nn.functional.upsample_bilinear": {f32, f64},
    "nn.functional.upsample_nearest": {f32, f64},
    # --- Data-dependent ops ---
    "nonzero": {b8, f32, f64, i32, i64},
    "nonzero_static": {b8, f32, f64, i32, i64},
    # --- RNG ops ---
    "normal": {f32, f64},
    ("normal", "in_place"): {f32, f64},
    ("normal", "number_mean"): {f32, f64},
    # --- Factory ops ---
    "ones": {b8, f32, f64, i32, i64},
    "ormqr": {f32, f64},
    "pca_lowrank": {f32, f64},
    "rand_like": {f32, f64},
    "randint": {f32, f64, i32, i64},
    "randint_like": {f32, f64, i32, i64},
    "randn": {f32, f64},
    "randn_like": {f32, f64},
    "repeat_interleave": {b8, f32, f64, i32, i64},
    ("round", "decimals_0"): {f32, f64},
    ("round", "decimals_3"): {f32, f64},
    ("round", "decimals_neg_3"): {f32, f64},
    "scalar_tensor": {b8, f32, f64, i32, i64},
    ("scatter_reduce", "amax"): {b8, f32, f64, i32, i64},
    ("scatter_reduce", "amin"): {b8, f32, f64, i32, i64},
    ("scatter_reduce", "mean"): {f32, f64, i32, i64},
    ("scatter_reduce", "prod"): {b8, f32, f64, i32, i64},
    ("scatter_reduce", "sum"): {b8, f32, f64, i32, i64},
    "searchsorted": {f32, f64, i32, i64},
    "signal.windows.bartlett": {f32, f64},
    "signal.windows.blackman": {f32, f64},
    "signal.windows.cosine": {f32, f64},
    "signal.windows.exponential": {f32, f64},
    "signal.windows.gaussian": {f32, f64},
    "signal.windows.general_cosine": {f32, f64},
    "signal.windows.general_hamming": {f32, f64},
    "signal.windows.hamming": {f32, f64},
    "signal.windows.hann": {f32, f64},
    "signal.windows.kaiser": {f32, f64},
    "signal.windows.nuttall": {f32, f64},
    ("softmax", "with_dtype"): {b8, f32, f64, i32, i64},
    "sparse.sampled_addmm": {f32, f64},
    "stft": {f32, f64},
    "svd": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensordot": {f32, f64, i32, i64},
    "to": {b8, f32, f64, i32, i64},
    "torch.ops.aten._safe_softmax.default": {b8, f32, f64, i32, i64},
    "tril_indices": {i32, i64},
    "triu_indices": {i32, i64},
    "uniform": {f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "zeros": {b8, f32, f64, i32, i64},
}

torchlite_expected_failures["cuda"] = {
    # --- Tensor property descriptors ---
    "H": {b8, f32, f64, i32, i64},
    "T": {b8, f32, f64, i32, i64},
    ("_segment_reduce", "lengths"): {f32, f64},
    ("_segment_reduce", "offsets"): {f32, f64},
    "_upsample_bilinear2d_aa": {f32, f64},
    "addbmm": {f32, f64},
    "addmm": {f32, f64},
    ("addmm", "decomposed"): {f32, f64},
    "addmv": {f32, f64},
    "allclose": {f32, f64},
    "arange": {f32, f64, i32, i64},
    "as_strided": {b8, f32, f64, i32, i64},
    "as_strided_copy": {b8, f32, f64, i32, i64},
    "as_strided_scatter": {b8, f32, f64, i32, i64},
    "baddbmm": {f32, f64},
    "bernoulli": {f32},
    "bincount": {i32, i64},
    "bucketize": {f32, f64, i32, i64},
    "cat": {b8, f32, f64, i32, i64},
    "cauchy": {f32, f64},
    "cholesky": {f32, f64},
    "combinations": {b8, f32, f64, i32, i64},
    ("div", "floor_rounding"): {f32, f64, i32, i64},
    ("div", "trunc_rounding"): {f32, f64, i32, i64},
    "exponential": {f32, f64},
    "eye": {b8, f32, f64, i32, i64},
    "fft.fft": {b8, f32, f64, i32, i64},
    "fft.fft2": {b8, f32, f64, i32, i64},
    "fft.fftn": {b8, f32, f64, i32, i64},
    "fft.hfft": {b8, f32, f64, i32, i64},
    "fft.hfft2": {b8, f32, f64, i32, i64},
    "fft.hfftn": {b8, f32, f64, i32, i64},
    "fft.ifft": {b8, f32, f64, i32, i64},
    "fft.ifft2": {b8, f32, f64, i32, i64},
    "fft.ifftn": {b8, f32, f64, i32, i64},
    "fft.ihfft": {b8, f32, f64, i32, i64},
    "fft.ihfft2": {b8, f32, f64, i32, i64},
    "fft.ihfftn": {b8, f32, f64, i32, i64},
    "fft.irfft": {b8, f32, f64, i32, i64},
    "fft.irfft2": {b8, f32, f64, i32, i64},
    "fft.irfftn": {b8, f32, f64, i32, i64},
    "fft.rfft": {b8, f32, f64, i32, i64},
    "fft.rfft2": {b8, f32, f64, i32, i64},
    "fft.rfftn": {b8, f32, f64, i32, i64},
    "fill": {b8, f32, f64, i32, i64},
    "flip": {b8, f32, f64, i32, i64},
    "full": {b8, f32, f64, i32, i64},
    "geometric": {f32, f64, i32, i64},
    "gradient": {f32, f64, i32, i64},
    "histc": {f32, f64, i32, i64},
    "index_add": {b8, f32, f64, i32, i64},
    "index_put": {b8, f32, f64, i32, i64},
    ("index_reduce", "amax"): {f32, f64, i32, i64},
    ("index_reduce", "amin"): {f32, f64, i32, i64},
    ("index_reduce", "mean"): {f32, f64, i32, i64},
    ("index_reduce", "prod"): {f32, f64, i32, i64},
    "jiterator_2inputs_2outputs": {b8, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f32, f64, i32, i64},
    "jiterator_binary": {b8, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f32, f64, i32, i64},
    "jiterator_unary": {b8, f32, f64, i32, i64},
    "linalg.cholesky": {f32, f64},
    "linalg.cholesky_ex": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.ldl_factor": {f32, f64},
    "linalg.ldl_factor_ex": {f32, f64},
    "linalg.ldl_solve": {f32, f64},
    "linalg.lstsq": {f32, f64},
    ("linalg.lstsq", "grad_oriented"): {f32, f64},
    "linalg.lu": {f32, f64},
    "linalg.lu_factor": {f32, f64},
    "linalg.lu_factor_ex": {f32, f64},
    "linalg.lu_solve": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    ("linalg.matrix_rank", "hermitian"): {f32, f64},
    "linalg.norm": {f32, f64},
    ("linalg.norm", "subgradients_at_zero"): {f32, f64},
    "linalg.pinv": {f32, f64},
    ("linalg.pinv", "hermitian"): {f32, f64},
    "linalg.solve_triangular": {f32, f64},
    "linalg.svd": {f32, f64},
    "linalg.tensorinv": {f32, f64},
    "linalg.tensorsolve": {f32, f64},
    "linalg.vander": {f32, f64, i32, i64},
    "linalg.vector_norm": {f32, f64},
    "linspace": {f32, f64, i32, i64},
    ("linspace", "tensor_overload"): {f32, f64, i32, i64},
    "log_normal": {f32, f64},
    ("log_softmax", "with_dtype"): {b8, f32, f64, i32, i64},
    "logspace": {f32, f64, i32, i64},
    ("logspace", "tensor_overload"): {f32, f64, i32, i64},
    "lu": {f32, f64},
    "lu_unpack": {f32, f64},
    "mH": {b8, f32, f64, i32, i64},
    "mT": {b8, f32, f64, i32, i64},
    "masked.amax": {f32, f64, i32, i64},
    "masked.amin": {f32, f64, i32, i64},
    "masked.argmax": {f32, f64, i32, i64},
    "masked.argmin": {f32, f64, i32, i64},
    "masked.cumprod": {f32, f64, i32, i64},
    "masked.cumsum": {f32, f64, i32, i64},
    "masked.log_softmax": {f32, f64},
    "masked.logaddexp": {f32, f64},
    "masked.logsumexp": {f32, f64, i32, i64},
    "masked.mean": {f32, f64},
    "masked.median": {f32, f64},
    "masked.norm": {f32, f64},
    "masked.prod": {b8, f32, f64, i32, i64},
    "masked.softmax": {f32, f64},
    "masked.softmin": {f32, f64},
    "masked.std": {f32, f64, i32, i64},
    "masked.sum": {b8, f32, f64, i32, i64},
    "masked.var": {f32, f64, i32, i64},
    "max_pool2d_with_indices_backward": {f32, f64},
    ("meshgrid", "list_of_tensors"): {b8, f32, f64, i32, i64},
    ("meshgrid", "variadic_tensors"): {b8, f32, f64, i32, i64},
    "multinomial": {f32, f64},
    "nn.functional.alpha_dropout": {f32, f64},
    "nn.functional.batch_norm": {f32, f64},
    ("nn.functional.batch_norm", "without_cudnn"): {f32, f64},
    "nn.functional.conv1d": {f32, f64},
    "nn.functional.conv2d": {f32, f64},
    "nn.functional.conv3d": {f32, f64},
    "nn.functional.conv_transpose1d": {f32, f64},
    "nn.functional.conv_transpose2d": {f32, f64},
    "nn.functional.conv_transpose3d": {f32, f64},
    "nn.functional.cosine_embedding_loss": {b8, f32, f64, i32, i64},
    "nn.functional.cosine_similarity": {f32, f64},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.dropout": {f32, f64},
    "nn.functional.dropout2d": {f32, f64},
    "nn.functional.dropout3d": {f32, f64},
    "nn.functional.embedding_bag": {f32, f64},
    ("nn.functional.feature_alpha_dropout", "with_train"): {f32, f64},
    ("nn.functional.feature_alpha_dropout", "without_train"): {b8, f32, f64, i32, i64},
    "nn.functional.fractional_max_pool2d": {f32, f64},
    "nn.functional.fractional_max_pool3d": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.gelu": {f32, f64},
    "nn.functional.grid_sample": {f32, f64},
    "nn.functional.group_norm": {f32, f64},
    "nn.functional.hardshrink": {f32, f64},
    "nn.functional.hardtanh": {f32, f64, i32, i64},
    "nn.functional.hinge_embedding_loss": {f32, f64},
    "nn.functional.huber_loss": {f32, f64},
    "nn.functional.instance_norm": {f32, f64},
    ("nn.functional.interpolate", "area"): {f32, f64},
    ("nn.functional.interpolate", "bicubic"): {f32, f64},
    ("nn.functional.interpolate", "bilinear"): {f32, f64},
    ("nn.functional.interpolate", "linear"): {f32, f64},
    ("nn.functional.interpolate", "nearest-exact"): {f32, f64},
    ("nn.functional.interpolate", "nearest"): {f32, f64},
    ("nn.functional.interpolate", "trilinear"): {f32, f64},
    "nn.functional.kl_div": {f32, f64},
    "nn.functional.layer_norm": {f32, f64},
    "nn.functional.local_response_norm": {f32, f64},
    "nn.functional.margin_ranking_loss": {f32, f64, i32, i64},
    "nn.functional.max_pool1d": {f32, f64},
    "nn.functional.max_pool2d": {f32, f64},
    "nn.functional.max_pool3d": {f32, f64},
    "nn.functional.max_unpool1d": {f32, f64},
    ("nn.functional.max_unpool1d", "grad"): {f32, f64},
    "nn.functional.max_unpool2d": {f32, f64},
    ("nn.functional.max_unpool2d", "grad"): {f32, f64},
    "nn.functional.max_unpool3d": {f32, f64},
    ("nn.functional.max_unpool3d", "grad"): {f32, f64},
    "nn.functional.multi_head_attention_forward": {f32, f64},
    "nn.functional.nll_loss": {f32, f64},
    "nn.functional.normalize": {f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pixel_shuffle": {b8, f32, f64, i32, i64},
    "nn.functional.pixel_unshuffle": {b8, f32, f64, i32, i64},
    "nn.functional.poisson_nll_loss": {f32, f64, i32, i64},
    "nn.functional.prelu": {f32, f64},
    "nn.functional.rms_norm": {f32, f64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.scaled_dot_product_attention": {f32, f64},
    ("nn.functional.softmin", "with_dtype"): {f32, f64, i32, i64},
    "nn.functional.softplus": {f32, f64},
    "nn.functional.softshrink": {f32, f64},
    "nn.functional.triplet_margin_loss": {f32, f64, i32, i64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64},
    "nn.functional.upsample_bilinear": {f32, f64},
    "nn.functional.upsample_nearest": {f32, f64},
    "nonzero": {b8, f32, f64, i32, i64},
    "normal": {f32, f64},
    ("normal", "in_place"): {f32, f64},
    ("normal", "number_mean"): {f32, f64},
    "ones": {b8, f32, f64, i32, i64},
    "ormqr": {f32, f64},
    "pca_lowrank": {f32, f64},
    "rand_like": {f32, f64},
    "randint": {f32, f64, i32, i64},
    "randint_like": {f32, f64, i32, i64},
    "randn": {f32, f64},
    "randn_like": {f32, f64},
    "repeat_interleave": {b8, f32, f64, i32, i64},
    ("round", "decimals_0"): {f32, f64},
    ("round", "decimals_3"): {f32, f64},
    ("round", "decimals_neg_3"): {f32, f64},
    "scalar_tensor": {b8, f32, f64, i32, i64},
    ("scatter_reduce", "amax"): {f32, f64, i32, i64},
    ("scatter_reduce", "amin"): {f32, f64, i32, i64},
    ("scatter_reduce", "mean"): {f32, f64, i32, i64},
    ("scatter_reduce", "prod"): {f32, f64, i32, i64},
    ("scatter_reduce", "sum"): {b8, f32, f64, i32, i64},
    "searchsorted": {f32, f64, i32, i64},
    "signal.windows.bartlett": {f32, f64},
    "signal.windows.blackman": {f32, f64},
    "signal.windows.cosine": {f32, f64},
    "signal.windows.exponential": {f32, f64},
    "signal.windows.gaussian": {f32, f64},
    "signal.windows.general_cosine": {f32, f64},
    "signal.windows.general_hamming": {f32, f64},
    "signal.windows.hamming": {f32, f64},
    "signal.windows.hann": {f32, f64},
    "signal.windows.kaiser": {f32, f64},
    "signal.windows.nuttall": {f32, f64},
    ("softmax", "with_dtype"): {b8, f32, f64, i32, i64},
    "sparse.sampled_addmm": {f32, f64},
    "stft": {f32, f64},
    "svd": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensordot": {f32, f64},
    "to": {b8, f32, f64, i32, i64},
    "torch.ops.aten._efficient_attention_forward": {f32},
    "torch.ops.aten._safe_softmax.default": {b8, f32, f64, i32, i64},
    "tril_indices": {i32, i64},
    "triu_indices": {i32, i64},
    "uniform": {f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "zeros": {b8, f32, f64, i32, i64},
}


def _get_skips_and_xfails(from_dict, xfails=True):
    retval = set()
    for device, d in from_dict.items():
        for op, dtypes in d.items():
            if isinstance(op, tuple):
                op, variant_name = op
            else:
                variant_name = ""
            retval.add((op, variant_name, device, tuple(dtypes), xfails))
    return retval


test_skips_or_fails = _get_skips_and_xfails(
    torchlite_skips, xfails=False
) | _get_skips_and_xfails(torchlite_expected_failures, xfails=True)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient test skip / xfail dictionaries.
#
# All forward-only failures are inherited as skips (can't test gradient if
# forward doesn't work). Additional gradient-specific failures go here.
# ──────────────────────────────────────────────────────────────────────────────

torchlite_grad_expected_failures = defaultdict(dict)
torchlite_grad_expected_failures["cpu"] = {}
torchlite_grad_expected_failures["cuda"] = {
    "H": {f32, f64},
    "T": {f32, f64},
    "linalg.lu": {f32, f64},
    "linalg.lu_factor": {f32, f64},
    "linalg.lu_factor_ex": {f32, f64},
    "lu": {f32, f64},
    "lu_unpack": {f32, f64},
    "mH": {f32, f64},
    "mT": {f32, f64},
    "nn.functional.multi_head_attention_forward": {f32, f64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64},
    "pca_lowrank": {f32, f64},
    "svd_lowrank": {f32, f64},
}

# Forward failures become gradient skips (only keep float dtypes)
_forward_fails_as_grad_skips = set()
for _dict in [torchlite_skips, torchlite_expected_failures]:
    for _dev, _ops_dict in _dict.items():
        for _op, _dtypes in _ops_dict.items():
            _float_dtypes = {d for d in _dtypes if d in (f32, f64)}
            if _float_dtypes:
                if isinstance(_op, tuple):
                    _op_name, _var = _op
                else:
                    _op_name, _var = _op, ""
                _forward_fails_as_grad_skips.add(
                    (_op_name, _var, _dev, tuple(_float_dtypes), False)
                )

grad_test_skips_or_fails = _forward_fails_as_grad_skips | _get_skips_and_xfails(
    torchlite_grad_expected_failures, xfails=True,
)


def _reduce_to_scalar(out):
    if isinstance(out, torch.Tensor):
        if out.is_floating_point():
            return out.sum()
        return None
    if isinstance(out, (tuple, list)):
        parts = [
            r.sum() for r in out
            if isinstance(r, torch.Tensor) and r.is_floating_point()
        ]
        return sum(parts) if parts else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Failure collection: run with PYTORCH_COLLECT_EXPECT=1 to auto-generate
# the skip/xfail dicts above.
# ──────────────────────────────────────────────────────────────────────────────
seen_failed = defaultdict(set)
seen_failed_grad = defaultdict(set)


def _print_dict(name, seen_dict):
    by_device = defaultdict(dict)
    for (device_type, op_key), dtypes in seen_dict.items():
        by_device[device_type][op_key] = dtypes
    for device_type in sorted(by_device):
        entries = by_device[device_type]
        print(f'\n{name}["{device_type}"] = {{')
        for op_key in sorted(entries, key=lambda k: k if isinstance(k, str) else k[0]):
            dtypes_str = ", ".join(sorted(str(d) for d in entries[op_key]))
            print(f"    {op_key!r}: {{{dtypes_str}}},")
        print("}")


def _print_seen():
    _print_dict("torchlite_expected_failures", seen_failed)
    if seen_failed_grad:
        _print_dict("torchlite_grad_expected_failures", seen_failed_grad)


if COLLECT_EXPECT:
    import atexit

    atexit.register(_print_seen)


def _make_collection_decorator(seen_dict):
    def decorator(fn):
        import functools
        import unittest

        @functools.wraps(fn)
        def inner(self, device, dtype, op):
            try:
                fn(self, device, dtype, op)
            except unittest.SkipTest:
                raise
            except Exception as e:
                if COLLECT_EXPECT:
                    variant = op.variant_test_name
                    op_key = op.name if not variant else (op.name, variant)
                    device_type = torch.device(device).type
                    seen_dict[device_type, op_key].add(dtype)
                raise e

        return inner
    return decorator


_collection_decorator = _make_collection_decorator(seen_failed)
_grad_collection_decorator = _make_collection_decorator(seen_failed_grad)


def _check_torchlite_op(self, fn, args, kwargs, *, assert_equal=True, atol=None, rtol=None, **_ignored):
    """Trace fn with torchlite and compare output against eager."""
    def _clone(inputs):
        return [
            x.clone().detach() if isinstance(x, torch.Tensor) else copy.deepcopy(x)
            for x in inputs
        ]

    ref_args = _clone(args)
    trace_args = _clone(args)
    replay_args = _clone(args)

    torch.manual_seed(0)
    expected = fn(*ref_args, **kwargs)

    gm = trace(fn, trace_args)

    torch.manual_seed(0)
    actual = gm(*replay_args, **kwargs)

    if atol is None:
        atol = 1e-4
    if rtol is None:
        rtol = 1e-4

    if assert_equal:
        self.assertEqual(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


class TestTorchliteOpInfo(TestCase):
    check_model = _check_torchlite_op

    @onlyNativeDeviceTypes
    @_ops(op_db[START:END])
    @skipOps("TestTorchliteOpInfo", "test_comprehensive", test_skips_or_fails)
    @_collection_decorator
    def test_comprehensive(self, device, dtype, op):
        torch._dynamo.reset()

        func = op.get_op()

        def fn(*args, **kwargs):
            return func(*args, **kwargs)

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        # Use only first sample for speed
        try:
            samples = [next(iter(samples))]
        except StopIteration:
            return

        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            self.check_model(
                fn,
                args,
                kwargs,
                assert_equal=True,
            )

    @onlyNativeDeviceTypes
    @_grad_ops(op_db[START:END])
    @skipOps("TestTorchliteOpInfo", "test_comprehensive_grad", grad_test_skips_or_fails)
    @_grad_collection_decorator
    def test_comprehensive_grad(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Op does not support autograd")

        torch._dynamo.reset()
        func = op.get_op()

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        try:
            sample = next(iter(samples))
        except StopIteration:
            return

        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs

        diff_indices = [
            i for i, a in enumerate(args)
            if isinstance(a, torch.Tensor) and a.requires_grad
        ]
        if not diff_indices:
            return

        def fn(*a):
            return func(*a, **kwargs)

        def _clone_args(inputs, with_grad):
            return [
                a.clone().detach().requires_grad_(with_grad and a.requires_grad)
                if isinstance(a, torch.Tensor)
                else copy.deepcopy(a)
                for a in inputs
            ]

        eager_args = _clone_args(args, with_grad=True)
        eager_out = fn(*eager_args)
        eager_loss = _reduce_to_scalar(eager_out)
        if eager_loss is None or not eager_loss.requires_grad:
            return

        eager_grads = torch.autograd.grad(
            eager_loss, [eager_args[i] for i in diff_indices],
            allow_unused=True,
        )

        trace_args = _clone_args(args, with_grad=False)
        replay_args = _clone_args(args, with_grad=True)

        gm = trace(fn, trace_args)
        traced_out = gm(*replay_args)
        traced_loss = _reduce_to_scalar(traced_out)
        if traced_loss is None or not traced_loss.requires_grad:
            self.fail("Traced output is not differentiable but eager output was")

        traced_grads = torch.autograd.grad(
            traced_loss, [replay_args[i] for i in diff_indices],
            allow_unused=True,
        )

        for idx, (eg, tg) in zip(diff_indices, zip(eager_grads, traced_grads)):
            if eg is None and tg is None:
                continue
            self.assertEqual(
                eg, tg, atol=1e-4, rtol=1e-4, equal_nan=True,
                msg=f"Gradient mismatch for arg {idx}",
            )


instantiate_device_type_tests(TestTorchliteOpInfo, globals())

if __name__ == "__main__":
    run_tests()
