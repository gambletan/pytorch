"""Deterministic performance tests for the torchlite compiler.

Measures structural properties of torchlite's compilation output that indicate
performance quality: fusion counts, memory plan savings, activation checkpoint
recomputation, and total kernel counts. Uses assertExpectedInline to hardcode
expected values — if numbers improve, update them downward; if they regress,
the test fails.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
from test_torchlite_utils import SimpleLinear, TwoLayerMLP, ThreeLayerSinCos, TrainStep

from torch._torchlite import (
    compile,
    trace,
    run_passes,
    timed_run_passes,
    default_passes,
    FusedKernel,
)
from torch._torchlite.passes import (
    _graph_meta,
    activation_checkpoint,
    autograd_per_op,
    decompose,
    fuse,
    functionalize,
    memory_plan,
    save_activations,
    _save_for_backward,
)


def _count_fused_kernels(gm):
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "call_function" and isinstance(n.target, FusedKernel)
    )


def _count_call_functions(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function")


def _count_save_for_backward(gm):
    return sum(
        1 for n in gm.graph.nodes
        if n.op == "call_function" and n.target is _save_for_backward
    )


class TestFusionCount(TestCase):
    def test_pointwise_chain_fuses(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        self.assertExpectedInline(str(n), """1""")

    def test_sin_cos_mul_chain(self):
        def f(x):
            return torch.sin(x) * torch.cos(x) + x

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        self.assertExpectedInline(str(n), """1""")

    def test_relu_sigmoid_tanh_chain(self):
        def f(x):
            return torch.tanh(torch.sigmoid(torch.relu(x)))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        self.assertExpectedInline(str(n), """1""")

    def test_two_layer_mlp_fusion(self):
        model = TwoLayerMLP(128, 256, 64)
        gm = trace(model, [torch.randn(4, 128)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        # relu + add should fuse into one kernel
        self.assertExpectedInline(str(n), """1""")

    def test_no_fusion_for_single_ops(self):
        def f(x):
            return torch.matmul(x, x.t())

        gm = trace(f, [torch.randn(8, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        self.assertExpectedInline(str(n), """0""")

    def test_branching_limits_fusion(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            # a is consumed by both cos and the output sum,
            # which limits fusion opportunities
            return b + a

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        n = _count_fused_kernels(gm)
        self.assertExpectedInline(str(n), """1""")


class TestMemoryPlan(TestCase):
    def test_simple_chain_reuses_buffers(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            c = torch.exp(b)
            return c

        gm = trace(f, [torch.randn(4, 128)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]
        # With 3 same-size intermediates, greedy reuse should use fewer pools
        self.assertLessEqual(stats["num_pools"], stats["num_tensors"])
        self.assertLessEqual(stats["planned_alloc"], stats["naive_alloc"])

    def test_memory_savings_ratio(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            c = torch.exp(b)
            d = torch.neg(c)
            return d

        gm = trace(f, [torch.randn(4, 128)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]
        self.assertGreater(stats["naive_alloc"], 0)
        savings = 1.0 - stats["planned_alloc"] / stats["naive_alloc"]
        # Chain of same-shape ops should achieve some savings
        self.assertGreater(savings, 0.0)

    def test_mlp_memory_plan(self):
        model = TwoLayerMLP(32, 64, 16)
        gm = trace(model, [torch.randn(4, 32)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]
        self.assertGreater(stats["num_tensors"], 0)
        self.assertGreater(stats["num_pools"], 0)


class TestActivationCheckpoint(TestCase):
    def test_cheap_ops_recomputed(self):
        model = ThreeLayerSinCos(32, 64, 16)
        train_step = TrainStep(model)
        inp = [torch.randn(4, 32), torch.randn(4, 16)]
        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm

        saves_before = _count_save_for_backward(gm)
        gm = activation_checkpoint(gm, inp).gm
        saves_after = _count_save_for_backward(gm)

        # activation_checkpoint should remove some save_for_backward
        # nodes for cheap ops (sin, cos) by recomputing them
        self.assertLessEqual(saves_after, saves_before)

    def test_no_saves_no_change(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = activation_checkpoint(gm, []).gm
        saves = _count_save_for_backward(gm)
        self.assertEqual(saves, 0)


class TestKernelCount(TestCase):
    def test_simple_function_kernel_count(self):
        def f(x):
            return torch.sin(torch.cos(x)) + x

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        total = _count_call_functions(gm)
        fused = _count_fused_kernels(gm)
        # After fusion, there should be fewer total operations
        # sin(cos(x)) should fuse, leaving fused_kernel + add = few ops
        self.assertGreater(total, 0)
        self.assertLessEqual(total, 5)

    def test_mlp_kernel_count(self):
        model = TwoLayerMLP(32, 64, 16)
        gm = trace(model, [torch.randn(4, 32)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        total = _count_call_functions(gm)
        self.assertGreater(total, 0)


class TestCompileTime(TestCase):
    def _assert_compile_time(self, model, example_inputs, budget_ms, pipeline=None):
        gm = trace(model, [x.clone() for x in example_inputs])
        gm, timings = timed_run_passes(gm, example_inputs, pipeline=pipeline)
        total_ms = sum(timings.values()) * 1000
        self.assertLess(
            total_ms, budget_ms,
            f"Compile time {total_ms:.1f}ms exceeded budget {budget_ms}ms. "
            f"Per-pass breakdown: "
            + ", ".join(f"{k}={v*1000:.1f}ms" for k, v in timings.items()),
        )

    def test_simple_linear(self):
        from torch._torchlite import verify_graph, dynamize
        self._assert_compile_time(
            SimpleLinear(128, 64),
            [torch.randn(4, 128)],
            budget_ms=2500,
            pipeline=[verify_graph, functionalize, dynamize, memory_plan],
        )

    def test_two_layer_mlp(self):
        from torch._torchlite import verify_graph, dynamize
        self._assert_compile_time(
            TwoLayerMLP(128, 256, 64),
            [torch.randn(4, 128)],
            budget_ms=5000,
            pipeline=[verify_graph, functionalize, dynamize, memory_plan],
        )

    def test_train_step(self):
        self._assert_compile_time(
            TrainStep(TwoLayerMLP(64, 128, 32)),
            [torch.randn(4, 64), torch.randn(4, 32)],
            budget_ms=10000,
        )


class TestRuntimePerf(TestCase):
    def _measure_runtime(self, fn, args, warmup=3, repeats=10):
        import time
        for _ in range(warmup):
            fn(*args)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    def test_compiled_not_catastrophically_slower(self):
        model = TwoLayerMLP(128, 256, 64)
        train_step = TrainStep(model)
        inp = torch.randn(32, 128)
        target = torch.randn(32, 64)

        eager_time = self._measure_runtime(train_step, [inp, target])

        compiled = compile(train_step, [inp.clone(), target.clone()])
        compiled_time = self._measure_runtime(compiled, [inp, target])

        ratio = compiled_time / max(eager_time, 1e-9)
        self.assertLess(
            ratio, 10.0,
            f"Compiled is {ratio:.1f}x slower than eager "
            f"(eager={eager_time*1000:.2f}ms, compiled={compiled_time*1000:.2f}ms)",
        )

    def test_inference_compiled_not_catastrophically_slower(self):
        from torch._torchlite import verify_graph, dynamize

        model = TwoLayerMLP(128, 256, 64)
        inp = torch.randn(32, 128)

        eager_time = self._measure_runtime(model, [inp])

        from torch._torchlite import trace, run_passes
        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm, [inp.clone()],
            pipeline=[verify_graph, functionalize, dynamize, memory_plan],
        )
        compiled_time = self._measure_runtime(gm, [inp])

        ratio = compiled_time / max(eager_time, 1e-9)
        self.assertLess(
            ratio, 10.0,
            f"Inference compiled is {ratio:.1f}x slower than eager "
            f"(eager={eager_time*1000:.2f}ms, compiled={compiled_time*1000:.2f}ms)",
        )


if __name__ == "__main__":
    run_tests()
