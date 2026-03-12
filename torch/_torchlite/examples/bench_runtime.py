"""Benchmark runtime performance: torchlite vs eager vs torch.compile.

Measures wall-clock inference and training throughput for several model
architectures, comparing torchlite, eager, and torch.compile. Also reports
compile-time for both torchlite and torch.compile.

Run with:
    python -m torch._torchlite.examples.bench_runtime

Environment variables:
    TORCHLITE_BENCH_WARMUP=N    Number of warmup iterations (default: 5)
    TORCHLITE_BENCH_ITERS=N     Number of timed iterations (default: 50)
    TORCHLITE_BENCH_DEVICE=dev  Device to benchmark on (default: cpu)
"""

import os
import time

import torch
import torch.nn as nn

from torch._torchlite import compile as torchlite_compile, trace
from torch._torchlite.passes import (
    dynamize,
    functionalize,
    memory_plan,
    verify_graph,
)


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        h = torch.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


class FourLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w3 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w4 = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        x = torch.relu(x @ self.w1)
        x = torch.relu(x @ self.w2)
        x = torch.relu(x @ self.w3)
        return x @ self.w4


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.wq = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wk = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wv = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.wo = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w1 = nn.Parameter(torch.randn(dim, dim * 4) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim * 4, dim) * 0.01)
        self.n_heads = n_heads
        self.dim = dim

    def forward(self, x):
        B, S, D = x.shape
        head_dim = D // self.n_heads

        q = (x @ self.wq).reshape(B, S, self.n_heads, head_dim).transpose(1, 2)
        k = (x @ self.wk).reshape(B, S, self.n_heads, head_dim).transpose(1, 2)
        v = (x @ self.wv).reshape(B, S, self.n_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + out @ self.wo

        h = torch.relu(x @ self.w1)
        x = x + h @ self.w2
        return x


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        if out.dim() == 3:
            out = out.mean(dim=1)
        return ((out - target) ** 2).mean()


INFERENCE_PIPELINE = [verify_graph, functionalize, dynamize, memory_plan]


def _time_fn(fn, args, warmup, iters, device):
    for _ in range(warmup):
        fn(*args)

    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / iters


def _time_compile(compile_fn, *args):
    t0 = time.perf_counter()
    result = compile_fn(*args)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def benchmark_inference(name, model_fn, input_fn, device, warmup, iters):
    torch.manual_seed(0)
    model = model_fn().to(device)
    inp = [x.to(device) for x in input_fn()]

    eager_ms = _time_fn(model, inp, warmup, iters, device)

    torch.manual_seed(0)
    model_tl = model_fn().to(device)
    inp_trace = [x.clone() for x in inp]
    gm, tl_compile_ms = _time_compile(trace, model_tl, inp_trace)
    tl_ms = _time_fn(gm, inp, warmup, iters, device)

    torch.manual_seed(0)
    model_tc = model_fn().to(device)
    tc_compiled, tc_compile_ms = _time_compile(
        lambda m: torch.compile(m, fullgraph=True), model_tc,
    )
    # torch.compile is lazy — first call triggers actual compilation
    tc_first_call_t0 = time.perf_counter()
    tc_compiled(*inp)
    tc_compile_ms += (time.perf_counter() - tc_first_call_t0) * 1000
    tc_ms = _time_fn(tc_compiled, inp, warmup, iters, device)

    tl_speedup = eager_ms / tl_ms if tl_ms > 0 else float("inf")
    tc_speedup = eager_ms / tc_ms if tc_ms > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print(f"  {name} (inference, {device})")
    print(f"{'=' * 70}")
    print(f"  {'Backend':<25s}  {'Runtime (ms)':<15s}  {'Speedup':<10s}  {'Compile (ms)'}")
    print(f"  {'-' * 65}")
    print(f"  {'eager':<25s}  {eager_ms:<15.3f}  {'1.00x':<10s}  {'n/a'}")
    print(f"  {'torchlite (trace)':<25s}  {tl_ms:<15.3f}  {tl_speedup:<10.2f}x {tl_compile_ms:.1f}")
    print(f"  {'torch.compile':<25s}  {tc_ms:<15.3f}  {tc_speedup:<10.2f}x {tc_compile_ms:.1f}")


def benchmark_training(name, model_fn, input_fn, device, warmup, iters):
    torch.manual_seed(0)
    model_eager = model_fn().to(device)
    inp = [x.to(device) for x in input_fn()]
    opt = torch.optim.SGD(model_eager.parameters(), lr=0.01)

    def eager_step():
        opt.zero_grad()
        loss = model_eager(*inp)
        loss.backward()
        opt.step()
        return loss

    eager_ms = _time_fn(eager_step, [], warmup, iters, device)

    torch.manual_seed(0)
    model_tl = model_fn().to(device)
    inp_tl = [x.to(device) for x in input_fn()]
    tl_compiled, tl_compile_ms = _time_compile(
        torchlite_compile, model_tl, inp_tl,
    )
    tl_ms = _time_fn(tl_compiled, inp_tl, warmup, iters, device)

    tl_speedup = eager_ms / tl_ms if tl_ms > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print(f"  {name} (training, {device})")
    print(f"{'=' * 70}")
    print(f"  {'Backend':<25s}  {'Runtime (ms)':<15s}  {'Speedup':<10s}  {'Compile (ms)'}")
    print(f"  {'-' * 65}")
    print(f"  {'eager':<25s}  {eager_ms:<15.3f}  {'1.00x':<10s}  {'n/a'}")
    print(f"  {'torchlite (compile)':<25s}  {tl_ms:<15.3f}  {tl_speedup:<10.2f}x {tl_compile_ms:.1f}")


INFERENCE_MODELS = {
    "TwoLayerMLP(256→512→128)": (
        lambda: TwoLayerMLP(256, 512, 128),
        lambda: [torch.randn(32, 256)],
    ),
    "FourLayerMLP(512)": (
        lambda: FourLayerMLP(512),
        lambda: [torch.randn(32, 512)],
    ),
    "TransformerBlock(256, 4heads)": (
        lambda: SimpleTransformerBlock(256, n_heads=4),
        lambda: [torch.randn(4, 64, 256)],
    ),
}

TRAINING_MODELS = {
    "TrainStep(TwoLayerMLP)": (
        lambda: TrainStep(TwoLayerMLP(256, 512, 128)),
        lambda: [torch.randn(32, 256), torch.randn(32, 128)],
    ),
    "TrainStep(FourLayerMLP)": (
        lambda: TrainStep(FourLayerMLP(256)),
        lambda: [torch.randn(32, 256), torch.randn(32, 256)],
    ),
}


def main():
    device = os.environ.get("TORCHLITE_BENCH_DEVICE", "cpu")
    warmup = int(os.environ.get("TORCHLITE_BENCH_WARMUP", "5"))
    iters = int(os.environ.get("TORCHLITE_BENCH_ITERS", "50"))

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Runtime benchmark: device={device}, warmup={warmup}, iters={iters}")

    print("\n\n--- INFERENCE ---")
    for name, (model_fn, input_fn) in INFERENCE_MODELS.items():
        try:
            benchmark_inference(name, model_fn, input_fn, device, warmup, iters)
        except Exception as e:
            print(f"\n  {name}: FAILED ({e})")

    print("\n\n--- TRAINING ---")
    for name, (model_fn, input_fn) in TRAINING_MODELS.items():
        try:
            benchmark_training(name, model_fn, input_fn, device, warmup, iters)
        except Exception as e:
            print(f"\n  {name}: FAILED ({e})")


if __name__ == "__main__":
    main()
