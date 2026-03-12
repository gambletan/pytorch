"""Sweep benchmark: torchlite (full inference pipeline) vs torch.compile.

Tests a range of model sizes and architectures to find where torchlite
is significantly slower than torch.compile. Uses the full inference
pipeline including matmul_epilogue + triton_lower + autotune.

Run with:
    python -m torch._torchlite.examples.bench_sweep
"""

import os
import time

import torch
import torch.nn as nn

from torch._torchlite.api import (
    codegen,
    inference_passes,
    run_passes,
    trace,
)


class LinearReLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TwoLayerMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class ThreeLayerMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, d)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class GeluMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)) * self.fc1(x))


class ResidualBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


def _time_fn(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


MODELS = [
    # (name, model_fn, batch, d_in)
    # Small sizes — kernel launch overhead matters
    ("LinearReLU 256->512", lambda: LinearReLU(256, 512), 32, 256),
    ("LinearReLU 512->1024", lambda: LinearReLU(512, 1024), 64, 512),

    # Medium sizes
    ("2layer d=512 h=1024", lambda: TwoLayerMLP(512, 1024), 64, 512),
    ("2layer d=1024 h=2048", lambda: TwoLayerMLP(1024, 2048), 128, 1024),
    ("2layer d=2048 h=4096", lambda: TwoLayerMLP(2048, 4096), 128, 2048),

    # Large sizes
    ("2layer d=4096 h=8192", lambda: TwoLayerMLP(4096, 8192), 128, 4096),
    ("2layer d=4096 h=16384", lambda: TwoLayerMLP(4096, 16384), 64, 4096),

    # Deeper networks
    ("3layer d=1024 h=2048", lambda: ThreeLayerMLP(1024, 2048), 128, 1024),
    ("3layer d=2048 h=4096", lambda: ThreeLayerMLP(2048, 4096), 128, 2048),

    # Different activations / patterns
    ("residual d=1024 h=2048", lambda: ResidualBlock(1024, 2048), 128, 1024),
    ("residual d=2048 h=4096", lambda: ResidualBlock(2048, 4096), 128, 2048),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    warmup = int(os.environ.get("TORCHLITE_BENCH_WARMUP", "10"))
    iters = int(os.environ.get("TORCHLITE_BENCH_ITERS", "100"))

    print(f"Sweep benchmark: warmup={warmup}, iters={iters}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    header = (
        f"{'Model':<30s}  {'Eager (ms)':<12s}  {'TorchLite':<12s}  "
        f"{'torch.compile':<14s}  {'TL/TC ratio':<12s}  {'Notes'}"
    )
    print(header)
    print("-" * len(header))

    for name, model_fn, batch, d_in in MODELS:
        try:
            torch.manual_seed(42)
            model = model_fn().cuda().eval()
            x = torch.randn(batch, d_in, device="cuda")

            # Eager
            with torch.no_grad():
                eager_ms = _time_fn(model, [x], warmup, iters)

            # TorchLite (full inference pipeline with triton_lower)
            torch.manual_seed(42)
            model_tl = model_fn().cuda().eval()
            with torch.no_grad():
                gm = trace(model_tl, [x])
                pipeline = inference_passes(gm, [x])
                gm = run_passes(gm, [x], pipeline=pipeline)
                fn_tl = codegen(gm, inference_codegen=True, example_inputs=[x])
                tl_ms = _time_fn(fn_tl, [x], warmup, iters)

            # torch.compile
            torch.manual_seed(42)
            model_tc = model_fn().cuda().eval()
            tc = torch.compile(model_tc, fullgraph=True)
            with torch.no_grad():
                tc(x)  # trigger compilation
                tc_ms = _time_fn(tc, [x], warmup, iters)

            ratio = tl_ms / tc_ms if tc_ms > 0 else float("inf")
            flag = ""
            if ratio > 1.15:
                flag = "<-- SLOWER"
            elif ratio < 0.85:
                flag = "<-- FASTER"

            print(
                f"{name:<30s}  {eager_ms:<12.3f}  {tl_ms:<12.3f}  "
                f"{tc_ms:<14.3f}  {ratio:<12.2f}  {flag}"
            )

            # Correctness check
            with torch.no_grad():
                out_eager = model(x)
                out_tl = fn_tl(x)
                if not torch.allclose(out_eager, out_tl, atol=0.05, rtol=0.05):
                    print(f"  WARNING: torchlite output differs! max_diff={( out_eager - out_tl).abs().max().item():.4f}")

        except Exception as e:
            print(f"{name:<30s}  FAILED: {e}")

    print()


if __name__ == "__main__":
    main()
