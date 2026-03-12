"""Triton code generation pass."""
import importlib.util
import os
import tempfile
from typing import Dict, List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _graph_meta,
    AddLayerNormKernel,
    AddRmsNormKernel,
    FusedKernel,
    MatmulEpilogueKernel,
    PassResult,
)


_MATMUL_AUTOTUNE_CONFIGS = [
    {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "num_warps": 2, "num_stages": 1},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
]


_TORCH_OP_MAP = {
    "sin": lambda acc, *a: torch.sin(acc),
    "cos": lambda acc, *a: torch.cos(acc),
    "exp": lambda acc, *a: torch.exp(acc),
    "log": lambda acc, *a: torch.log(acc),
    "abs": lambda acc, *a: torch.abs(acc),
    "neg": lambda acc, *a: -acc,
    "sqrt": lambda acc, *a: torch.sqrt(acc),
    "rsqrt": lambda acc, *a: torch.rsqrt(acc),
    "sigmoid": lambda acc, *a: torch.sigmoid(acc),
    "tanh": lambda acc, *a: torch.tanh(acc),
    "relu": lambda acc, *a: torch.relu(acc),
    "silu": lambda acc, *a: torch.nn.functional.silu(acc),
    "gelu": lambda acc, *a: torch.nn.functional.gelu(acc),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "reciprocal": lambda acc, *a: 1.0 / acc,
}

_TORCH_UNARY_INPLACE_OP_MAP = {
    "relu": lambda acc: torch.relu_(acc),
    "silu": lambda acc: torch.nn.functional.silu(acc, inplace=True),
    "sigmoid": lambda acc: torch.sigmoid_(acc),
    "tanh": lambda acc: torch.tanh_(acc),
    "neg": lambda acc: acc.neg_(),
    "abs": lambda acc: acc.abs_(),
}


def _apply_epilogue_inplace(acc, epilogue_ops, input_t, weight_t, bias, extras):
    """Apply epilogue ops to acc, preferring in-place ops to avoid allocation."""
    for op in epilogue_ops:
        if op.op_name in _TORCH_UNARY_INPLACE_OP_MAP and len(op.args) == 1 and op.args[0][0] in ("acc", "tmp"):
            acc = _TORCH_UNARY_INPLACE_OP_MAP[op.op_name](acc)
            continue
        if op.op_name in ("add", "mul") and len(op.args) == 2:
            def _resolve(arg):
                tag = arg[0]
                if tag in ("acc", "tmp"):
                    return acc
                elif tag == "input":
                    inputs_list = [input_t, weight_t]
                    if bias is not None:
                        inputs_list.append(bias)
                    return inputs_list[arg[1]]
                elif tag == "const":
                    return arg[1]
                elif tag == "extra":
                    return extras[arg[1]]
            a, b = _resolve(op.args[0]), _resolve(op.args[1])
            if op.op_name == "add":
                if a is acc:
                    acc = acc.add_(b)
                elif b is acc:
                    acc = acc.add_(a)
                else:
                    acc = a + b
            else:
                if a is acc:
                    acc = acc.mul_(b)
                elif b is acc:
                    acc = acc.mul_(a)
                else:
                    acc = a * b
            continue
        fn = _TORCH_OP_MAP.get(op.op_name)
        if fn is None:
            continue
        arg_vals = []
        for arg in op.args:
            tag = arg[0]
            if tag in ("acc", "tmp"):
                arg_vals.append(acc)
            elif tag == "input":
                inputs_list = [input_t, weight_t]
                if bias is not None:
                    inputs_list.append(bias)
                arg_vals.append(inputs_list[arg[1]])
            elif tag == "const":
                arg_vals.append(arg[1])
            elif tag == "extra":
                arg_vals.append(extras[arg[1]])
        acc = fn(*arg_vals)
    return acc


_TRITON_OP_MAP = {
    "sin": ("tl.math.sin(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "cos": ("tl.math.cos(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "exp": ("tl.math.exp(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "log": ("tl.math.log(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "abs": ("tl.abs({})", 1),
    "neg": ("-({})", 1),
    "sqrt": ("tl.math.sqrt(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "rsqrt": ("tl.math.rsqrt(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "sigmoid": ("tl.sigmoid(({0}).to(tl.float32)).to(({0}).dtype)", 1),
    "tanh": ("(2.0 * tl.sigmoid((2.0 * ({0})).to(tl.float32)) - 1.0).to(({0}).dtype)", 1),
    "relu": ("tl.maximum({}, 0.0)", 1),
    "silu": ("(({0}) * tl.sigmoid(({0}).to(tl.float32))).to(({0}).dtype)", 1),
    "gelu": ("(0.5 * {0} * (1.0 + (2.0 * tl.sigmoid((2.0 * 0.7978845608028654 * ({0} + 0.044715 * {0} * {0} * {0})).to(tl.float32)) - 1.0))).to(({0}).dtype)", 1),
    "add": ("({} + {})", 2),
    "sub": ("({} - {})", 2),
    "mul": ("({} * {})", 2),
    "div": ("({} / {})", 2),
    "where": ("tl.where({}, {}, {})", 3),
    "reciprocal": ("(1.0 / {})", 1),
}


def triton_codegen(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Generate Triton GPU kernel source code for fused ops in the graph.

    Walks the graph looking for FusedKernel nodes (created by the fuse pass)
    and emits Triton JIT kernel code for each one. The generated code is
    stored in the graph's metadata under the key "triton_code".
    """
    kernels = [
        _generate_kernel_source(node.target)
        for node in gm.graph.nodes
        if node.op == "call_function" and isinstance(node.target, FusedKernel)
    ]

    code = "\n\n\n".join(kernels) + "\n" if kernels else "# No fused kernels found\n"
    _graph_meta(gm.graph)["triton_code"] = code
    return PassResult(gm=gm)


def _generate_kernel_source(kernel: FusedKernel) -> str:
    """Generate Triton kernel source for a single FusedKernel.

    For inputs whose shape differs from the output shape, broadcast-aware
    index expressions are emitted. A 1D input of size N (last dimension of
    a 2D output [M, N]) uses `offs % N`; a column-vector [M, 1] uses
    `offs // N`.  These cover the two broadcast patterns that appear in
    RMSNorm-style fusions where rsqrt[M,1] and weight[N] are multiplied
    with x[M,N].
    """
    out_shape = kernel.shape or []
    in_ptrs = [f"in_ptr{i}" for i in range(kernel.n_inputs)]
    params = ", ".join(
        in_ptrs + ["out_ptr", "n_elements", "BLOCK_SIZE: tl.constexpr = 1024"]
    )

    lines = [
        "@triton.jit",
        f"def {kernel.name}({params}):",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offs < n_elements",
        "",
    ]

    n_out = len(out_shape)
    out_n = out_shape[-1] if n_out >= 1 else 1

    input_shapes = kernel.input_shapes or [None] * kernel.n_inputs
    for i in range(kernel.n_inputs):
        in_shape = input_shapes[i] if i < len(input_shapes) else None
        if in_shape is None or list(in_shape) == list(out_shape):
            idx_expr = "offs"
            load_mask = "mask"
        elif n_out >= 2 and len(in_shape) == 1 and in_shape[0] == out_shape[-1]:
            # 1D weight/bias of size D broadcast along all leading dims:
            # flat index for the last dimension is offs % D.  Works for
            # both 2D [M, D] and 3D [B, S, D] output tensors.
            idx_expr = f"offs % {out_n}"
            load_mask = f"(offs % {out_n}) < {out_n}"
        elif n_out >= 2 and in_shape[-1] == 1 and list(in_shape[:-1]) == list(out_shape[:-1]):
            # Column-vector shaped [*, 1] broadcast over the last dim:
            # drop the last dimension by integer-dividing out D.  Covers
            # both [M, 1] in 2D and [B, S, 1] in 3D.
            idx_expr = f"offs // {out_n}"
            n_rows = 1
            for s in in_shape[:-1]:
                n_rows *= s
            load_mask = f"(offs // {out_n}) < {n_rows}"
        else:
            idx_expr = "offs"
            load_mask = "mask"
        lines.append(f"    x{i} = tl.load(in_ptr{i} + {idx_expr}, mask={load_mask})")
    lines.append("")

    val_map: Dict[tuple, str] = {}
    for i in range(kernel.n_inputs):
        val_map[("input", i)] = f"x{i}"

    for tmp_idx, op in enumerate(kernel.ops):
        entry = _TRITON_OP_MAP.get(op.op_name)
        if entry is None:
            continue
        template, nargs = entry

        arg_vars = []
        for arg in op.args:
            key = (arg[0], arg[1])
            if key in val_map:
                arg_vars.append(val_map[key])
            elif arg[0] == "const":
                arg_vars.append(str(arg[1]))
            else:
                arg_vars.append("???")

        result = f"tmp{tmp_idx}"

        if nargs == 1 and arg_vars:
            expr = template.format(arg_vars[0])
        elif nargs == 2 and len(arg_vars) >= 2:
            expr = template.format(arg_vars[0], arg_vars[1])
        elif nargs == 3 and len(arg_vars) >= 3:
            expr = template.format(arg_vars[0], arg_vars[1], arg_vars[2])
        else:
            expr = f"# {op.op_name}({', '.join(arg_vars)})"

        lines.append(f"    {result} = {expr}")
        val_map[("tmp", tmp_idx)] = result

    lines.append("")
    if kernel.ops:
        lines.append(
            f"    tl.store(out_ptr + offs, tmp{len(kernel.ops) - 1}, mask=mask)"
        )
    else:
        lines.append("    pass")

    return "\n".join(lines)


class _TritonKernelModule(torch.nn.Module):
    """Wraps a compiled Triton kernel as an nn.Module for use as a call_module node."""
    def __init__(self, triton_fn, shape, numel, dtype):
        super().__init__()
        self.triton_fn = triton_fn
        self.shape = shape
        self.numel = numel
        self.dtype = dtype

    def forward(self, *inputs):
        # The generated kernel assumes contiguous layout for broadcast index
        # calculations (offs % N, offs // N).  Sliced or permuted tensors
        # can have non-unit strides that violate this, so force contiguous.
        inputs = tuple(
            x.contiguous() if isinstance(x, torch.Tensor) and not x.is_contiguous()
            else x
            for x in inputs
        )
        out = torch.empty(self.shape, dtype=self.dtype, device=inputs[0].device)
        grid = ((self.numel + 1023) // 1024,)
        self.triton_fn[grid](*inputs, out, self.numel)
        return out


def _generate_add_rms_norm_source(kernel: AddRmsNormKernel) -> str:
    """Generate a Triton reduction kernel for fused add + rms_norm.

    When has_add=True (default), each program handles one row:
    1. Loads elements of a and b, computes c = a + b
    2. Stores c to add_out (the add result is needed by downstream residual)
    3. Computes variance = mean(c^2) over the row
    4. Normalizes: out = c * rsqrt(variance + eps) * weight
    5. Stores out to norm_out

    When has_add=False, the kernel is a standalone rms_norm:
    1. Loads elements of x
    2. Computes variance = mean(x^2) over the row
    3. Normalizes: out = x * rsqrt(variance + eps) * weight
    4. Stores out to norm_out
    """
    if not kernel.has_add:
        lines = [
            "@triton.jit",
            f"def {kernel.name}(",
            "    x_ptr, weight_ptr,",
            "    norm_out_ptr,",
            "    n_rows, n_cols: tl.constexpr, eps,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    row_idx = tl.program_id(0)",
            "    col_offs = tl.arange(0, BLOCK_SIZE)",
            "    mask = col_offs < n_cols",
            "    row_start = row_idx * n_cols",
            "",
            "    c = tl.load(x_ptr + row_start + col_offs, mask=mask)",
            "",
            "    c_f32 = c.to(tl.float32)",
            "    variance = tl.sum(c_f32 * c_f32, axis=0) / n_cols",
            "    rrms = tl.math.rsqrt(variance + eps)",
            "",
            "    weight = tl.load(weight_ptr + col_offs, mask=mask)",
            "    out = (c_f32 * rrms).to(c.dtype) * weight",
            "    tl.store(norm_out_ptr + row_start + col_offs, out, mask=mask)",
        ]
        return "\n".join(lines)

    lines = [
        "@triton.jit",
        f"def {kernel.name}(",
        "    a_ptr, b_ptr, weight_ptr,",
        "    add_out_ptr, norm_out_ptr,",
        "    n_rows, n_cols: tl.constexpr, eps,",
        "    BLOCK_SIZE: tl.constexpr,",
        "):",
        "    row_idx = tl.program_id(0)",
        "    col_offs = tl.arange(0, BLOCK_SIZE)",
        "    mask = col_offs < n_cols",
        "    row_start = row_idx * n_cols",
        "",
        "    a = tl.load(a_ptr + row_start + col_offs, mask=mask)",
        "    b = tl.load(b_ptr + row_start + col_offs, mask=mask)",
        "    c = a + b",
        "",
        "    tl.store(add_out_ptr + row_start + col_offs, c, mask=mask)",
        "",
        "    c_f32 = c.to(tl.float32)",
        "    variance = tl.sum(c_f32 * c_f32, axis=0) / n_cols",
        "    rrms = tl.math.rsqrt(variance + eps)",
        "",
        "    weight = tl.load(weight_ptr + col_offs, mask=mask)",
        "    out = (c_f32 * rrms).to(c.dtype) * weight",
        "    tl.store(norm_out_ptr + row_start + col_offs, out, mask=mask)",
    ]
    return "\n".join(lines)


class _AddRmsNormModule(torch.nn.Module):
    """Wraps a fused add+rms_norm or standalone rms_norm Triton kernel.

    When has_add=True: takes (a, b, weight), returns (add_result, norm_result).
    When has_add=False: takes (x, weight), returns norm_result.
    """
    def __init__(self, triton_fn, shape, norm_dim, eps, dtype, has_add=True):
        super().__init__()
        self.triton_fn = triton_fn
        self.shape = shape
        self.norm_dim = norm_dim
        self.eps = eps
        self.dtype = dtype
        self.has_add = has_add
        self.n_rows = 1
        for s in shape[:-1]:
            self.n_rows *= s
        self.block_size = 1
        while self.block_size < norm_dim:
            self.block_size *= 2

    def _launch(self, a, b, weight, add_out, norm_out):
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        grid = (self.n_rows,)
        self.triton_fn[grid](
            a, b, weight, add_out, norm_out,
            self.n_rows, self.norm_dim, self.eps,
            BLOCK_SIZE=self.block_size,
        )

    def _launch_standalone(self, x, weight, norm_out):
        if not x.is_contiguous():
            x = x.contiguous()
        grid = (self.n_rows,)
        self.triton_fn[grid](
            x, weight, norm_out,
            self.n_rows, self.norm_dim, self.eps,
            BLOCK_SIZE=self.block_size,
        )

    def forward(self, *args):
        if self.has_add:
            a, b, weight = args
            add_out = torch.empty(self.shape, dtype=self.dtype, device=a.device)
            norm_out = torch.empty(self.shape, dtype=self.dtype, device=a.device)
            self._launch(a, b, weight, add_out, norm_out)
            return add_out, norm_out
        else:
            x, weight = args
            norm_out = torch.empty(self.shape, dtype=self.dtype, device=x.device)
            self._launch_standalone(x, weight, norm_out)
            return norm_out

    def forward_into_bufs(self, a, b, weight, add_buf, norm_buf):
        self._launch(a, b, weight, add_buf, norm_buf)
        return add_buf, norm_buf


def _compile_add_rms_norm_kernels(
    kernels: List[AddRmsNormKernel],
) -> Dict[str, object]:
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
    ]
    for kernel in kernels:
        lines.append(_generate_add_rms_norm_source(kernel))
        lines.append("")
        lines.append("")

    fd, path = tempfile.mkstemp(
        suffix=".py", prefix="torchlite_add_rms_norm_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
        spec = importlib.util.spec_from_file_location(
            "_torchlite_add_rms_norm", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(path)

    return {k.name: getattr(mod, k.name) for k in kernels}


def _generate_add_layer_norm_source(kernel: AddLayerNormKernel) -> str:
    """Generate a Triton reduction kernel for fused add + layer_norm.

    LayerNorm computes mean and variance (unlike RMSNorm which only uses
    variance/RMS), so this kernel has two reduction passes:
    1. mean = sum(x) / n_cols
    2. variance = sum((x - mean)^2) / n_cols
    3. out = (x - mean) / sqrt(variance + eps) * weight [+ bias]
    """
    bias_param = ", bias_ptr" if kernel.has_norm_bias else ""

    if not kernel.has_add:
        lines = [
            "@triton.jit",
            f"def {kernel.name}(",
            f"    x_ptr, weight_ptr{bias_param},",
            "    norm_out_ptr,",
            "    n_rows, n_cols: tl.constexpr, eps,",
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    row_idx = tl.program_id(0)",
            "    col_offs = tl.arange(0, BLOCK_SIZE)",
            "    mask = col_offs < n_cols",
            "    row_start = row_idx * n_cols",
            "",
            "    c = tl.load(x_ptr + row_start + col_offs, mask=mask, other=0.0)",
            "",
            "    c_f32 = c.to(tl.float32)",
            "    mean = tl.sum(c_f32, axis=0) / n_cols",
            "    centered = c_f32 - mean",
            "    variance = tl.sum(centered * centered, axis=0) / n_cols",
            "    rstd = tl.math.rsqrt(variance + eps)",
            "",
            "    weight = tl.load(weight_ptr + col_offs, mask=mask)",
            "    out = (centered * rstd).to(c.dtype) * weight",
        ]
        if kernel.has_norm_bias:
            lines += [
                "    bias = tl.load(bias_ptr + col_offs, mask=mask)",
                "    out = out + bias",
            ]
        lines.append(
            "    tl.store(norm_out_ptr + row_start + col_offs, out, mask=mask)"
        )
        return "\n".join(lines)

    lines = [
        "@triton.jit",
        f"def {kernel.name}(",
        f"    a_ptr, b_ptr, weight_ptr{bias_param},",
        "    add_out_ptr, norm_out_ptr,",
        "    n_rows, n_cols: tl.constexpr, eps,",
        "    BLOCK_SIZE: tl.constexpr,",
        "):",
        "    row_idx = tl.program_id(0)",
        "    col_offs = tl.arange(0, BLOCK_SIZE)",
        "    mask = col_offs < n_cols",
        "    row_start = row_idx * n_cols",
        "",
        "    a = tl.load(a_ptr + row_start + col_offs, mask=mask, other=0.0)",
        "    b = tl.load(b_ptr + row_start + col_offs, mask=mask, other=0.0)",
        "    c = a + b",
        "",
        "    tl.store(add_out_ptr + row_start + col_offs, c, mask=mask)",
        "",
        "    c_f32 = c.to(tl.float32)",
        "    mean = tl.sum(c_f32, axis=0) / n_cols",
        "    centered = c_f32 - mean",
        "    variance = tl.sum(centered * centered, axis=0) / n_cols",
        "    rstd = tl.math.rsqrt(variance + eps)",
        "",
        "    weight = tl.load(weight_ptr + col_offs, mask=mask)",
        "    out = (centered * rstd).to(c.dtype) * weight",
    ]
    if kernel.has_norm_bias:
        lines += [
            "    bias = tl.load(bias_ptr + col_offs, mask=mask)",
            "    out = out + bias",
        ]
    lines.append(
        "    tl.store(norm_out_ptr + row_start + col_offs, out, mask=mask)"
    )
    return "\n".join(lines)


class _AddLayerNormModule(torch.nn.Module):
    """Wraps a fused add+layer_norm or standalone layer_norm Triton kernel.

    When has_add=True: takes (a, b, weight[, bias]), returns (add_result, norm_result).
    When has_add=False: takes (x, weight[, bias]), returns norm_result.
    """
    def __init__(self, triton_fn, shape, norm_dim, eps, dtype, has_add=True, has_norm_bias=False):
        super().__init__()
        self.triton_fn = triton_fn
        self.shape = shape
        self.norm_dim = norm_dim
        self.eps = eps
        self.dtype = dtype
        self.has_add = has_add
        self.has_norm_bias = has_norm_bias
        self.n_rows = 1
        for s in shape[:-1]:
            self.n_rows *= s
        self.block_size = 1
        while self.block_size < norm_dim:
            self.block_size *= 2

    def _launch(self, a, b, weight, norm_bias, add_out, norm_out):
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        grid = (self.n_rows,)
        args = [a, b, weight]
        if self.has_norm_bias:
            args.append(norm_bias)
        args.extend([add_out, norm_out, self.n_rows, self.norm_dim, self.eps])
        self.triton_fn[grid](*args, BLOCK_SIZE=self.block_size)

    def _launch_standalone(self, x, weight, norm_bias, norm_out):
        if not x.is_contiguous():
            x = x.contiguous()
        grid = (self.n_rows,)
        args = [x, weight]
        if self.has_norm_bias:
            args.append(norm_bias)
        args.extend([norm_out, self.n_rows, self.norm_dim, self.eps])
        self.triton_fn[grid](*args, BLOCK_SIZE=self.block_size)

    def forward(self, *args):
        if self.has_add:
            if self.has_norm_bias:
                a, b, weight, norm_bias = args
            else:
                a, b, weight = args
                norm_bias = None
            add_out = torch.empty(self.shape, dtype=self.dtype, device=a.device)
            norm_out = torch.empty(self.shape, dtype=self.dtype, device=a.device)
            self._launch(a, b, weight, norm_bias, add_out, norm_out)
            return add_out, norm_out
        else:
            if self.has_norm_bias:
                x, weight, norm_bias = args
            else:
                x, weight = args
                norm_bias = None
            norm_out = torch.empty(self.shape, dtype=self.dtype, device=x.device)
            self._launch_standalone(x, weight, norm_bias, norm_out)
            return norm_out

    def forward_into_bufs(self, *args_and_bufs):
        """Like forward() but writes outputs into pre-allocated buffers."""
        if self.has_norm_bias:
            a, b, weight, norm_bias, add_buf, norm_buf = args_and_bufs
        else:
            a, b, weight, add_buf, norm_buf = args_and_bufs
            norm_bias = None
        self._launch(a, b, weight, norm_bias, add_buf, norm_buf)
        return add_buf, norm_buf


def _compile_add_layer_norm_kernels(
    kernels: List[AddLayerNormKernel],
) -> Dict[str, object]:
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
    ]
    for kernel in kernels:
        lines.append(_generate_add_layer_norm_source(kernel))
        lines.append("")
        lines.append("")

    fd, path = tempfile.mkstemp(
        suffix=".py", prefix="torchlite_add_layer_norm_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
        spec = importlib.util.spec_from_file_location(
            "_torchlite_add_layer_norm", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(path)

    return {k.name: getattr(mod, k.name) for k in kernels}


def _compile_triton_kernels(kernels: List[FusedKernel]) -> Dict[str, object]:
    """Write all kernels to a temp file, import the module, return name->fn map.

    Triton's @jit decorator calls inspect.getsourcelines, which requires
    the decorated function to live in a real file on disk. We write all
    kernels for a graph into a single temp module to amortize the cost.
    """
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
    ]
    for kernel in kernels:
        lines.append(_generate_kernel_source(kernel))
        lines.append("")
        lines.append("")

    fd, path = tempfile.mkstemp(suffix=".py", prefix="torchlite_kernels_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))

        spec = importlib.util.spec_from_file_location("_torchlite_kernels", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(path)

    return {k.name: getattr(mod, k.name) for k in kernels}


def _compile_matmul_kernels(kernels: List[MatmulEpilogueKernel]) -> Dict[str, object]:
    """Write matmul epilogue kernels to a temp file, import, return name->fn map.

    Deduplicates kernels that generate identical Triton source (same
    epilogue ops, same has_bias, same extra_shapes structure) so they
    share a single @autotune function. This ensures the autotune cache
    is populated once and reused across all instances with the same
    structure, avoiding repeated autotuning at runtime.
    """
    unique_sources: Dict[str, str] = {}
    kernel_to_canonical: Dict[str, str] = {}

    for kernel in kernels:
        source = _generate_matmul_epilogue_source(kernel)
        source_lines = source.split("\n")
        body = "\n".join(source_lines[3:])

        canonical_name = None
        for cname, cbody in unique_sources.items():
            if cbody == body:
                canonical_name = cname
                break

        if canonical_name is None:
            unique_sources[kernel.name] = body
            kernel_to_canonical[kernel.name] = kernel.name
        else:
            kernel_to_canonical[kernel.name] = canonical_name

    lines = [
        "import triton",
        "import triton.language as tl",
        "",
    ]
    for kernel in kernels:
        if kernel_to_canonical[kernel.name] == kernel.name:
            lines.append(_generate_matmul_epilogue_source(kernel))
            lines.append("")
            lines.append("")

    fd, path = tempfile.mkstemp(suffix=".py", prefix="torchlite_matmul_kernels_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))

        spec = importlib.util.spec_from_file_location("_torchlite_matmul_kernels", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(path)

    return {
        k.name: getattr(mod, kernel_to_canonical[k.name])
        for k in kernels
    }


def _generate_matmul_epilogue_source(kernel: MatmulEpilogueKernel) -> str:
    """Generate a 2D tiled Triton kernel for matmul + epilogue ops.

    Emits a kernel that tiles the M and N dimensions, accumulates via
    tl.dot over the K dimension, then applies the epilogue pointwise
    ops to the accumulator before storing.
    """
    n_extra = sum(1 for op in kernel.epilogue_ops for a in op.args if a[0] == "extra")
    extra_params = ", ".join(f"extra_ptr{i}" for i in range(n_extra))
    bias_param = "bias_ptr, " if kernel.has_bias else ""
    extra_sep = ", " if extra_params else ""

    extra_stride_params = []
    for i in range(n_extra):
        if i < len(kernel.extra_shapes) and len(kernel.extra_shapes[i]) >= 2:
            extra_stride_params.append(f"stride_extra{i}_m, stride_extra{i}_n")
    extra_stride_str = ", ".join(extra_stride_params)
    extra_stride_sep = ", " if extra_stride_str else ""

    params = (
        f"input_ptr, weight_t_ptr, {bias_param}"
        f"out_ptr{extra_sep}{extra_params}, "
        f"M, N, K, "
        f"stride_am, stride_ak, "
        f"stride_bk, stride_bn, "
        f"stride_cm, stride_cn"
        f"{extra_stride_sep}{extra_stride_str}, "
        f"BLOCK_M: tl.constexpr, "
        f"BLOCK_N: tl.constexpr, "
        f"BLOCK_K: tl.constexpr"
    )

    configs_src = ", ".join(
        f'triton.Config({{"BLOCK_M": {c["BLOCK_M"]}, "BLOCK_N": {c["BLOCK_N"]}, '
        f'"BLOCK_K": {c["BLOCK_K"]}}}, num_warps={c["num_warps"]}, '
        f'num_stages={c["num_stages"]})'
        for c in _MATMUL_AUTOTUNE_CONFIGS
    )

    lines = [
        f"@triton.autotune(configs=[{configs_src}], key=['M', 'N', 'K'])",
        "@triton.jit",
        f"def {kernel.name}({params}):",
        "    pid = tl.program_id(0)",
        "    num_n_blocks = tl.cdiv(N, BLOCK_N)",
        "    pid_m = pid // num_n_blocks",
        "    pid_n = pid % num_n_blocks",
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        "    offs_k = tl.arange(0, BLOCK_K)",
        f"    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
        "    for k_start in range(0, K, BLOCK_K):",
        "        k_offs = k_start + offs_k",
        "        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)",
        "        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)",
        "        a = tl.load(input_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)",
        "        b = tl.load(weight_t_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)",
        "        acc += tl.dot(a, b)",
    ]

    if kernel.has_bias:
        lines += [
            "    bias_mask = offs_n < N",
            "    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)",
            "    acc = acc + bias[None, :]",
        ]

    val_map = {}
    extra_idx = 0
    for tmp_idx, op in enumerate(kernel.epilogue_ops):
        entry = _TRITON_OP_MAP.get(op.op_name)
        if entry is None:
            continue
        template, nargs = entry

        arg_vars = []
        for arg in op.args:
            tag = arg[0]
            if tag == "acc":
                arg_vars.append("acc")
            elif tag == "tmp":
                var = val_map.get(("tmp", arg[1]))
                arg_vars.append(var if var else "acc")
            elif tag == "const":
                arg_vars.append(str(arg[1]))
            elif tag == "extra":
                ei = arg[1]
                is_2d = (
                    ei < len(kernel.extra_shapes)
                    and len(kernel.extra_shapes[ei]) >= 2
                )
                if is_2d:
                    lines.append(
                        f"    extra{extra_idx}_mask = "
                        f"(offs_m[:, None] < M) & (offs_n[None, :] < N)"
                    )
                    lines.append(
                        f"    extra{extra_idx} = tl.load("
                        f"extra_ptr{ei} "
                        f"+ offs_m[:, None] * stride_extra{ei}_m "
                        f"+ offs_n[None, :] * stride_extra{ei}_n, "
                        f"mask=extra{extra_idx}_mask, other=0.0)"
                    )
                    arg_vars.append(f"extra{extra_idx}")
                else:
                    lines.append(
                        f"    extra{extra_idx}_mask = offs_n < N"
                    )
                    lines.append(
                        f"    extra{extra_idx} = tl.load("
                        f"extra_ptr{ei} + offs_n, "
                        f"mask=extra{extra_idx}_mask, other=0.0)"
                    )
                    arg_vars.append(f"extra{extra_idx}[None, :]")
                extra_idx += 1
            elif tag == "input":
                lines.append(
                    f"    input_ep{extra_idx}_mask = "
                    f"(offs_m[:, None] < M) & (offs_n[None, :] < N)"
                )
                lines.append(
                    f"    input_ep{extra_idx} = tl.load("
                    f"input_ptr + offs_m[:, None] * stride_am "
                    f"+ offs_n[None, :] * stride_ak, "
                    f"mask=input_ep{extra_idx}_mask, other=0.0)"
                )
                arg_vars.append(f"input_ep{extra_idx}")
                extra_idx += 1

        result_var = f"ep_tmp{tmp_idx}"
        if nargs == 1 and arg_vars:
            expr = template.format(arg_vars[0])
        elif nargs == 2 and len(arg_vars) >= 2:
            expr = template.format(arg_vars[0], arg_vars[1])
        elif nargs == 3 and len(arg_vars) >= 3:
            expr = template.format(arg_vars[0], arg_vars[1], arg_vars[2])
        else:
            expr = f"# {op.op_name}({', '.join(arg_vars)})"

        lines.append(f"    {result_var} = {expr}")
        lines.append(f"    acc = {result_var}")
        val_map[("tmp", tmp_idx)] = result_var

    lines += [
        "    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
        "    tl.store(out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=out_mask)",
    ]

    return "\n".join(lines)


def _generate_epilogue_pointwise_source(name, epilogue_ops, has_bias, extra_shapes=None):
    """Generate a Triton 1D pointwise kernel for bias + epilogue ops.

    This kernel applies bias addition and epilogue operations (relu, silu, etc.)
    in a single fused pass, avoiding separate kernel launches for each op.
    Used when cuBLAS wins the matmul benchmark but we still want fused epilogue.

    When extra_shapes is provided, the kernel accepts additional tensor pointer
    arguments for binary ops (e.g. add(acc, residual)). Each extra tensor is
    loaded with the same flat indexing as acc (same shape) or broadcast via
    offs % N (1D bias-like shape).
    """
    extra_shapes = extra_shapes or []
    n_extras = len(extra_shapes)
    extra_ptr_params = ", ".join(f"extra_ptr{i}" for i in range(n_extras))
    extra_sep = ", " if extra_ptr_params else ""

    if has_bias:
        params = (
            f"acc_ptr, bias_ptr{extra_sep}{extra_ptr_params}, "
            f"out_ptr, N, n_elements, BLOCK_SIZE: tl.constexpr = 1024"
        )
    else:
        params = (
            f"acc_ptr{extra_sep}{extra_ptr_params}, "
            f"out_ptr, N, n_elements, BLOCK_SIZE: tl.constexpr = 1024"
        )

    lines = [
        "@triton.jit",
        f"def {name}({params}):",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offs < n_elements",
        "",
        "    acc = tl.load(acc_ptr + offs, mask=mask)",
    ]

    if has_bias:
        lines += [
            "    bias = tl.load(bias_ptr + (offs % N), mask=mask)",
            "    acc = acc + bias",
        ]

    for i in range(n_extras):
        eshape = extra_shapes[i] if i < len(extra_shapes) else ()
        if len(eshape) == 1:
            lines.append(f"    extra{i} = tl.load(extra_ptr{i} + (offs % N), mask=mask)")
        else:
            lines.append(f"    extra{i} = tl.load(extra_ptr{i} + offs, mask=mask)")

    for tmp_idx, op in enumerate(epilogue_ops):
        entry = _TRITON_OP_MAP.get(op.op_name)
        if entry is None:
            continue
        template, nargs = entry
        if nargs == 1:
            expr = template.format("acc")
            lines.append(f"    acc = {expr}")
        elif nargs == 2:
            arg_vars = []
            for arg in op.args:
                tag = arg[0]
                if tag in ("acc", "tmp"):
                    arg_vars.append("acc")
                elif tag == "extra":
                    arg_vars.append(f"extra{arg[1]}")
                elif tag == "const":
                    arg_vars.append(str(arg[1]))
            if len(arg_vars) == 2:
                expr = template.format(arg_vars[0], arg_vars[1])
                lines.append(f"    acc = {expr}")

    lines += [
        "",
        "    tl.store(out_ptr + offs, acc, mask=mask)",
    ]
    return "\n".join(lines)


_epilogue_kernel_cache: Dict[str, object] = {}


def _get_epilogue_kernel(name, epilogue_ops, has_bias, extra_shapes=None):
    """Compile (or retrieve from cache) a Triton pointwise epilogue kernel."""
    cache_key = name
    if cache_key in _epilogue_kernel_cache:
        return _epilogue_kernel_cache[cache_key]

    source = _generate_epilogue_pointwise_source(name, epilogue_ops, has_bias, extra_shapes)
    code_lines = [
        "import triton",
        "import triton.language as tl",
        "",
        source,
    ]
    fd, path = tempfile.mkstemp(suffix=".py", prefix="torchlite_epilogue_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(code_lines))
        spec = importlib.util.spec_from_file_location("_torchlite_epilogue", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(path)

    fn = getattr(mod, name)
    _epilogue_kernel_cache[cache_key] = fn
    return fn


def _cublas_matmul_epilogue(input_t, weight_t, bias, extras, epilogue_ops, dtype, out_shape=None):
    """Execute matmul via cuBLAS then apply epilogue ops eagerly.

    When out_shape is provided (e.g. [B, S, N] for a 3D batch), the 2D
    mm result [M, N] is reshaped to out_shape before applying the epilogue
    so that broadcast-sensitive ops (silu, mul with [B, S, 1]) see the
    correct shape.  The final result is returned in out_shape form.
    """
    if bias is not None and not epilogue_ops:
        acc = torch.addmm(bias, input_t, weight_t)
    else:
        acc = torch.mm(input_t, weight_t)
        if bias is not None:
            acc.add_(bias)
    if out_shape is not None and list(acc.shape) != list(out_shape):
        acc = acc.view(out_shape)
    acc = _apply_epilogue_inplace(acc, epilogue_ops, input_t, weight_t, bias, extras)
    return acc


def _is_fusable_ep_op(op):
    if op.op_name not in _TRITON_OP_MAP:
        return False
    if len(op.args) == 1 and op.args[0][0] in ("acc", "tmp"):
        return True
    if len(op.args) == 2:
        tags = {a[0] for a in op.args}
        return bool(tags & {"acc", "tmp"}) and tags <= {"acc", "tmp", "extra", "const"}
    return False


class _TritonMatmulModule(torch.nn.Module):
    """Wraps a fused matmul+epilogue Triton kernel with cuBLAS fallback.

    On first invocation, benchmarks the Triton fused kernel against
    cuBLAS + eager epilogue. Caches the winner (keyed by M, N, K)
    and uses it for all subsequent calls. This ensures we never pay
    more than cuBLAS for the matmul portion.
    """

    _backend_cache: Dict[tuple, str] = {}

    def __init__(self, triton_fn, M, N, K, dtype, has_bias, n_extra, epilogue_ops, extra_shapes=None, out_shape=None):
        super().__init__()
        self.triton_fn = triton_fn
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.has_bias = has_bias
        self.n_extra = n_extra
        self.epilogue_ops = epilogue_ops
        self.extra_shapes = extra_shapes or []
        self.out_shape = out_shape
        self._use_cublas = None
        self._epilogue_fn = None
        # Check if epilogue is purely unary (no extra inputs, no binary ops
        # with non-acc args) — these can be fused into a Triton pointwise
        # kernel to avoid separate eager kernel launches when cuBLAS wins.
        self._can_fuse_epilogue = all(
            _is_fusable_ep_op(op) for op in epilogue_ops
        )

    def _run_triton(self, input_t, weight_t, bias, extras):
        import triton

        M, N, K = self.M, self.N, self.K
        out = torch.empty((M, N), dtype=self.dtype, device=input_t.device)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        # The Triton kernel uses 2D strides (stride_m, stride_n) for extra
        # inputs.  When inputs arrive as ND tensors (e.g. [B, S, N] from a
        # 3D batch), we flatten leading dimensions to produce a contiguous
        # 2D view [M, N] before computing strides so memory addresses match.
        flat_extras = [
            ext.reshape(M, -1) if ext.ndim > 2 else ext
            for ext in extras
        ]
        args = [input_t, weight_t]
        if bias is not None:
            args.append(bias)
        args.append(out)
        args.extend(flat_extras)
        args.extend([
            M, N, K,
            input_t.stride(0), input_t.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            out.stride(0), out.stride(1),
        ])
        for i, ext in enumerate(flat_extras):
            if i < len(self.extra_shapes) and len(self.extra_shapes[i]) >= 2:
                args.extend([ext.stride(0), ext.stride(1)])
        self.triton_fn[grid](*args)
        if self.out_shape is not None and list(out.shape) != list(self.out_shape):
            out = out.view(self.out_shape)
        return out

    def _benchmark_backends(self, input_t, weight_t, bias, extras):
        epilogue_sig = tuple(
            (op.op_name, len(op.args), any(a[0] == "extra" for a in op.args))
            for op in self.epilogue_ops
        )
        cache_key = (self.M, self.N, self.K, self.dtype, self.has_bias, epilogue_sig)
        cached = _TritonMatmulModule._backend_cache.get(cache_key)
        if cached is not None:
            self._use_cublas = cached == "cublas"
            return

        bench_buf = torch.empty(
            (self.M, self.N), dtype=self.dtype, device=input_t.device
        )

        # When the epilogue can be fused into a single Triton pointwise kernel,
        # benchmark cuBLAS with that fused epilogue — not with separate eager
        # ops. Otherwise the benchmark unfairly penalizes cuBLAS by measuring
        # extra kernel launches that won't happen in production.
        use_fused_ep = self._can_fuse_epilogue and self.epilogue_ops
        if use_fused_ep:
            self._ensure_epilogue_fn()

        def _run_cublas_inplace():
            torch.ops.aten.mm.out(input_t, weight_t, out=bench_buf)
            acc = bench_buf
            if self.out_shape is not None and list(acc.shape) != list(self.out_shape):
                acc = acc.view(self.out_shape)
            if use_fused_ep:
                self._cublas_with_triton_epilogue(acc, bias, extras)
            else:
                if bias is not None:
                    bench_buf.add_(bias)
                _apply_epilogue_inplace(acc, self.epilogue_ops, input_t, weight_t, bias, extras)

        for _ in range(5):
            self._run_triton(input_t, weight_t, bias, extras)
            _run_cublas_inplace()
        torch.cuda.synchronize()

        # Per-call CUDA event timing avoids CPU dispatch gaps inflating
        # the measured time for the faster backend. Batch timing (single
        # start/end around N calls) includes GPU idle time when the CPU
        # can't submit the next kernel fast enough, which systematically
        # penalises cuBLAS (faster GPU, more Python overhead per call).
        n_bench = 25
        triton_times = []
        for _ in range(n_bench):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self._run_triton(input_t, weight_t, bias, extras)
            e.record()
            torch.cuda.synchronize()
            triton_times.append(s.elapsed_time(e))

        cublas_times = []
        for _ in range(n_bench):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _run_cublas_inplace()
            e.record()
            torch.cuda.synchronize()
            cublas_times.append(s.elapsed_time(e))

        triton_ms = sorted(triton_times)[n_bench // 2]
        cublas_ms = sorted(cublas_times)[n_bench // 2]

        winner = "cublas" if cublas_ms <= triton_ms else "triton"
        self._use_cublas = winner == "cublas"
        _TritonMatmulModule._backend_cache[cache_key] = winner

    def _run_triton_into_buf(self, buf, input_t, weight_t, bias, extras):
        """Like _run_triton but writes output into a pre-allocated buf."""
        import triton

        M, N, K = self.M, self.N, self.K
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        flat_extras = [
            ext.reshape(M, -1) if ext.ndim > 2 else ext
            for ext in extras
        ]
        args = [input_t, weight_t]
        if bias is not None:
            args.append(bias)
        args.append(buf)
        args.extend(flat_extras)
        args.extend([
            M, N, K,
            input_t.stride(0), input_t.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            buf.stride(0), buf.stride(1),
        ])
        for i, ext in enumerate(flat_extras):
            if i < len(self.extra_shapes) and len(self.extra_shapes[i]) >= 2:
                args.extend([ext.stride(0), ext.stride(1)])
        self.triton_fn[grid](*args)
        out = buf
        if self.out_shape is not None and list(out.shape) != list(self.out_shape):
            out = out.view(self.out_shape)
        return out

    def _ensure_epilogue_fn(self):
        """Lazily compile the Triton pointwise epilogue kernel."""
        if self._epilogue_fn is not None:
            return
        epilogue_sig = "_".join(op.op_name for op in self.epilogue_ops[:3])
        n_extra = len(self.extra_shapes)
        extra_suffix = f"_e{n_extra}" if n_extra else ""
        name = f"epilogue_{epilogue_sig}_{self.M}_{self.N}{extra_suffix}"
        self._epilogue_fn = _get_epilogue_kernel(
            name, self.epilogue_ops, self.has_bias, self.extra_shapes,
        )

    def _cublas_with_triton_epilogue(self, acc, bias, extras=None):
        """Apply bias + epilogue ops via a single Triton pointwise kernel.

        Writes in-place back to acc (out_ptr == acc_ptr). This is safe
        because the epilogue is purely pointwise: each thread reads acc[i],
        computes, and writes out[i] with no cross-element dependencies.
        """
        self._ensure_epilogue_fn()
        numel = acc.numel()
        grid = ((numel + 1023) // 1024,)
        extras = extras or []
        flat_extras = [
            ext.reshape(-1) if ext.ndim > 1 and list(ext.shape) != [1] else ext
            for ext in extras
        ]
        if self.has_bias:
            self._epilogue_fn[grid](acc, bias, *flat_extras, acc, self.N, numel)
        else:
            self._epilogue_fn[grid](acc, *flat_extras, acc, self.N, numel)
        return acc

    def _forward_into_buf(self, buf, input_t, weight_t, *rest):
        """Like forward() but writes matmul result into pre-allocated buf.

        The buf tensor is used for the matmul output, avoiding per-call
        allocation. If buf is not 2D [M, N] (e.g. it's a 3D pool buffer
        shaped [B, S, N]), it is viewed to [M, N] for the matmul and then
        reshaped back to out_shape if needed. Epilogue ops are applied
        after the matmul as usual.
        """
        bias = rest[0] if self.has_bias else None
        extras = list(rest[1:] if self.has_bias else rest)

        if self._use_cublas is None:
            if input_t.is_cuda:
                self._benchmark_backends(input_t, weight_t, bias, extras)
            else:
                self._use_cublas = True

        matmul_buf = buf.view(self.M, self.N) if list(buf.shape) != [self.M, self.N] else buf

        if self._use_cublas:
            if self._can_fuse_epilogue and self.epilogue_ops and input_t.is_cuda:
                torch.ops.aten.mm.out(input_t, weight_t, out=matmul_buf)
                acc = matmul_buf
                if self.out_shape is not None and list(acc.shape) != list(self.out_shape):
                    acc = acc.view(self.out_shape)
                return self._cublas_with_triton_epilogue(acc, bias, extras)
            if bias is not None and not self.epilogue_ops:
                torch.ops.aten.addmm.out(bias, input_t, weight_t, out=matmul_buf)
            else:
                torch.ops.aten.mm.out(input_t, weight_t, out=matmul_buf)
                if bias is not None:
                    matmul_buf.add_(bias)
            acc = matmul_buf
            if self.out_shape is not None and list(acc.shape) != list(self.out_shape):
                acc = acc.view(self.out_shape)
            return _apply_epilogue_inplace(acc, self.epilogue_ops, input_t, weight_t, bias, extras)
        return self._run_triton_into_buf(matmul_buf, input_t, weight_t, bias, extras)

    def forward(self, input_t, weight_t, *rest):
        bias = rest[0] if self.has_bias else None
        extras = list(rest[1:] if self.has_bias else rest)

        if self._use_cublas is None:
            if input_t.is_cuda:
                self._benchmark_backends(input_t, weight_t, bias, extras)
            else:
                self._use_cublas = True

        if self._use_cublas:
            if self._can_fuse_epilogue and self.epilogue_ops and input_t.is_cuda:
                acc = torch.mm(input_t, weight_t)
                if self.out_shape is not None and list(acc.shape) != list(self.out_shape):
                    acc = acc.view(self.out_shape)
                return self._cublas_with_triton_epilogue(acc, bias, extras)
            return _cublas_matmul_epilogue(
                input_t, weight_t, bias, extras,
                self.epilogue_ops, self.dtype, self.out_shape,
            )
        return self._run_triton(input_t, weight_t, bias, extras)


def triton_lower(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """JIT-compile FusedKernel nodes into callable Triton kernels.

    Unlike triton_codegen which only generates source strings, this pass
    actually compiles the Triton kernels and replaces FusedKernel node
    targets with callable wrappers so the graph remains executable.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        return PassResult(gm=gm)

    fused_kernels = []
    fused_nodes = []
    matmul_kernels = []
    matmul_nodes = []
    add_rms_kernels = []
    add_rms_nodes = []
    add_ln_kernels = []
    add_ln_nodes = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if isinstance(node.target, FusedKernel):
            fused_kernels.append(node.target)
            fused_nodes.append(node)
        elif isinstance(node.target, MatmulEpilogueKernel):
            matmul_kernels.append(node.target)
            matmul_nodes.append(node)
        elif isinstance(node.target, AddRmsNormKernel):
            add_rms_kernels.append(node.target)
            add_rms_nodes.append(node)
        elif isinstance(node.target, AddLayerNormKernel):
            add_ln_kernels.append(node.target)
            add_ln_nodes.append(node)

    if not fused_kernels and not matmul_kernels and not add_rms_kernels and not add_ln_kernels:
        return PassResult(gm=gm)

    if fused_kernels:
        name_to_fn = _compile_triton_kernels(fused_kernels)
        for node, kernel in zip(fused_nodes, fused_kernels):
            triton_fn = name_to_fn[kernel.name]
            shape = kernel.shape or []
            numel = 1
            for s in shape:
                numel *= s
            dtype = node.meta.get("dtype", torch.float32)
            mod = _TritonKernelModule(triton_fn, shape, numel, dtype)
            mod_name = f"_triton_{kernel.name}"
            gm.add_module(mod_name, mod)
            node.op = "call_module"
            node.target = mod_name

    if matmul_kernels:
        matmul_name_to_fn = _compile_matmul_kernels(matmul_kernels)
        for node, kernel in zip(matmul_nodes, matmul_kernels):
            triton_fn = matmul_name_to_fn[kernel.name]
            n_extra = sum(
                1 for op in kernel.epilogue_ops
                for a in op.args if a[0] == "extra"
            )
            mod = _TritonMatmulModule(
                triton_fn, kernel.M, kernel.N, kernel.K,
                kernel.dtype, kernel.has_bias, n_extra,
                kernel.epilogue_ops, kernel.extra_shapes,
                out_shape=kernel.out_shape,
            )
            mod_name = f"_triton_{kernel.name}"
            gm.add_module(mod_name, mod)
            node.op = "call_module"
            node.target = mod_name

    if add_rms_kernels:
        add_rms_name_to_fn = _compile_add_rms_norm_kernels(add_rms_kernels)
        for node, kernel in zip(add_rms_nodes, add_rms_kernels):
            triton_fn = add_rms_name_to_fn[kernel.name]
            mod = _AddRmsNormModule(
                triton_fn, kernel.shape, kernel.norm_dim,
                kernel.eps, kernel.dtype, kernel.has_add,
            )
            mod_name = f"_triton_{kernel.name}"
            gm.add_module(mod_name, mod)
            node.op = "call_module"
            node.target = mod_name

    if add_ln_kernels:
        add_ln_name_to_fn = _compile_add_layer_norm_kernels(add_ln_kernels)
        for node, kernel in zip(add_ln_nodes, add_ln_kernels):
            triton_fn = add_ln_name_to_fn[kernel.name]
            mod = _AddLayerNormModule(
                triton_fn, kernel.shape, kernel.norm_dim,
                kernel.eps, kernel.dtype, kernel.has_add,
                kernel.has_norm_bias,
            )
            mod_name = f"_triton_{kernel.name}"
            gm.add_module(mod_name, mod)
            node.op = "call_module"
            node.target = mod_name

    gm.graph.lint()
    gm.recompile()
    return PassResult(gm=gm)

