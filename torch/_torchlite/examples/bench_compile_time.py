"""Benchmark compile-time for torchlite passes.

Measures wall-clock time for trace() and each individual pass across
several model architectures. Run with:

    python -m torch._torchlite.examples.bench_compile_time

Set TORCHLITE_BENCH_REPEAT=N to average over N runs (default: 3).
"""

import os
import time

import torch
import torch.nn as nn

from torch._torchlite import trace, timed_run_passes
from torch._torchlite.passes import (
    dynamize,
    functionalize,
    memory_plan,
    verify_graph,
)


class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.weight + self.bias


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


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        return ((out - target) ** 2).mean()


MODELS = {
    "SimpleLinear(64→32)": (
        lambda: SimpleLinear(64, 32),
        lambda: [torch.randn(8, 64)],
    ),
    "TwoLayerMLP(128→256→64)": (
        lambda: TwoLayerMLP(128, 256, 64),
        lambda: [torch.randn(8, 128)],
    ),
    "FourLayerMLP(256)": (
        lambda: FourLayerMLP(256),
        lambda: [torch.randn(8, 256)],
    ),
}

TRAIN_MODELS = {
    "TrainStep(TwoLayerMLP)": (
        lambda: TrainStep(TwoLayerMLP(64, 128, 32)),
        lambda: [torch.randn(4, 64), torch.randn(4, 32)],
    ),
    "TrainStep(FourLayerMLP)": (
        lambda: TrainStep(FourLayerMLP(128)),
        lambda: [torch.randn(4, 128), torch.randn(4, 128)],
    ),
}


INFERENCE_PIPELINE = [verify_graph, functionalize, dynamize, memory_plan]


def benchmark_model(name, model_fn, inputs_fn, repeats, pipeline=None):
    trace_times = []
    all_pass_times = []

    for _ in range(repeats):
        model = model_fn()
        inputs = inputs_fn()

        t0 = time.perf_counter()
        gm = trace(model, [x.clone() for x in inputs])
        trace_times.append(time.perf_counter() - t0)

        _, timings = timed_run_passes(gm, inputs, pipeline=pipeline)
        all_pass_times.append(timings)

    avg_trace = sum(trace_times) / len(trace_times)
    avg_pass_times = {}
    for key in all_pass_times[0]:
        avg_pass_times[key] = sum(t[key] for t in all_pass_times) / len(all_pass_times)

    total = avg_trace + sum(avg_pass_times.values())

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'trace()':<30s}  {avg_trace * 1000:8.2f} ms")
    for pass_name, t in avg_pass_times.items():
        print(f"  {pass_name:<30s}  {t * 1000:8.2f} ms")
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL':<30s}  {total * 1000:8.2f} ms")


def benchmark_torch_compile(name, model_fn, inputs_fn, repeats):
    times = []
    for _ in range(repeats):
        torch._dynamo.reset()
        model = model_fn()
        inputs = inputs_fn()
        t0 = time.perf_counter()
        compiled = torch.compile(model, fullgraph=True)
        compiled(*inputs)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"  {'torch.compile (total)':<30s}  {avg * 1000:8.2f} ms")


def main():
    repeats = int(os.environ.get("TORCHLITE_BENCH_REPEAT", "3"))
    print(f"Compile-time benchmark (averaging {repeats} runs)")

    print("\n--- Inference Models ---")
    for name, (model_fn, inputs_fn) in MODELS.items():
        benchmark_model(name, model_fn, inputs_fn, repeats, pipeline=INFERENCE_PIPELINE)
        benchmark_torch_compile(name, model_fn, inputs_fn, repeats)

    print("\n--- Training Models (full pipeline) ---")
    for name, (model_fn, inputs_fn) in TRAIN_MODELS.items():
        benchmark_model(name, model_fn, inputs_fn, repeats)


if __name__ == "__main__":
    main()
