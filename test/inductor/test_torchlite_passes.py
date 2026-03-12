"""Unit tests for individual torchlite graph passes.

Each test creates a small model, traces it, runs a single pass, and verifies
the expected graph property or compares output against eager execution.
"""

import operator
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from test_torchlite_utils import TrainStep, TwoLayerMLP

from torch._torchlite import trace, run_passes, compile, precompile_save, precompile_load
from torch._torchlite.passes import (
    _graph_meta,
    _save_for_backward,
    activation_checkpoint,
    autograd_per_op,
    cudagraph_partition,
    decompose,
    dynamize,
    functionalize,
    fuse,
    FusedKernel,
    MatmulEpilogueKernel,
    matmul_epilogue,
    memory_plan,
    save_activations,
    normalize,
    optimizer,
    rng_functionalize,
    triton_codegen,
    verify_graph,
)
from torch._torchlite.ops import (
    _save_rng_state,
    _load_rng_state,
    param_update,
    adamw_step,
)


class TestNormalize(TestCase):
    def test_dunders_normalized(self):
        def f(x, y):
            return x + y

        gm = trace(f, [torch.randn(4), torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIs(node.target, torch.add)

    def test_tensor_method_normalized(self):
        def f(x):
            return x.abs()

        gm = trace(f, [torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIs(node.target, torch.abs)


class TestVerifyGraph(TestCase):
    def test_valid_graph_passes(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        gm = trace(f, [torch.randn(4)])
        result = verify_graph(gm, [])
        self.assertIsNotNone(result.gm)

    def test_detects_non_callable_target(self):
        gm = trace(lambda x: x + 1, [torch.randn(4)])
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.target = "not_callable"
                break
        with self.assertRaises(ValueError):
            verify_graph(gm, [])


class TestFunctionalize(TestCase):
    def test_inplace_add_removed(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        gm = trace(f, [torch.randn(4)])
        gm = functionalize(gm, []).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = getattr(node.target, "__name__", "")
                # No in-place ops should remain in the main body
                # (copy_back nodes are allowed at the end)
                if name.endswith("_") and not name.startswith("__"):
                    if node.target is not torch.Tensor.copy_:
                        self.fail(f"In-place op {name} not removed by functionalize")

    def test_functionalize_preserves_output(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        inp = torch.randn(4)
        expected = f(inp.clone())

        gm = trace(f, [inp.clone()])
        gm = functionalize(gm, []).gm
        actual = gm(inp.clone())
        self.assertEqual(actual, expected)

    def test_multiple_inplace_ops(self):
        def f(x):
            y = x.clone()
            y.mul_(2.0)
            y.add_(3.0)
            return y

        inp = torch.randn(4)
        expected = f(inp.clone())

        gm = trace(f, [inp.clone()])
        gm = functionalize(gm, []).gm
        actual = gm(inp.clone())
        self.assertEqual(actual, expected)


class TestDynamize(TestCase):
    def test_inserts_size_nodes(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = dynamize(gm, [torch.randn(4, 8)]).gm

        has_size = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.Tensor.size:
                has_size = True
                break
        self.assertTrue(has_size)

    def test_dynamic_batch_dim(self):
        def f(x):
            return x * 2

        inp4 = torch.randn(4, 8)
        inp8 = torch.randn(8, 8)

        gm = trace(f, [inp4])
        gm = dynamize(gm, [inp4]).gm

        out4 = gm(inp4)
        self.assertEqual(out4.shape, (4, 8))
        out8 = gm(inp8)
        self.assertEqual(out8.shape, (8, 8))

    def test_no_dynamic_dims_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm_before = str(gm.graph)
        gm = dynamize(gm, [torch.randn(4, 8)], dynamic_dims={}).gm
        # With empty dynamic_dims, no size nodes should be inserted
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.Tensor.size:
                self.fail("Size nodes inserted when dynamic_dims is empty")


class TestAutogradPerOp(TestCase):
    def test_backward_nodes_added(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm

        has_backward = False
        for node in gm.graph.nodes:
            if node.meta.get("phase") == "backward":
                has_backward = True
                break
        self.assertTrue(has_backward)

    def test_no_params_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm_orig = str(gm.graph)
        gm = autograd_per_op(gm, [torch.randn(4)]).gm
        # No params means no backward nodes should be added
        for node in gm.graph.nodes:
            self.assertNotEqual(node.meta.get("phase"), "backward")


class TestSaveActivations(TestCase):
    def test_save_for_backward_inserted(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm

        saves = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _save_for_backward
        )
        self.assertGreater(saves, 0)

    def test_no_backward_noop(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm = save_activations(gm, []).gm
        saves = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function" and n.target is _save_for_backward
        )
        self.assertEqual(saves, 0)


class TestActivationCheckpoint(TestCase):
    def test_recompute_nodes_added(self):
        class SinCosModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 4) * 0.01)

            def forward(self, x):
                h = torch.sin(x @ self.w)
                return torch.cos(h)

        model = SinCosModel()
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = activation_checkpoint(gm, inp).gm

        has_recompute = False
        for node in gm.graph.nodes:
            if node.meta.get("phase") == "recompute":
                has_recompute = True
                break
        self.assertTrue(has_recompute)


class TestDecompose(TestCase):
    def test_decompose_preserves_output(self):
        def f(x):
            return torch.nn.functional.gelu(x)

        inp = torch.randn(4, 8)
        expected = f(inp)

        gm = trace(f, [inp.clone()])
        gm = decompose(gm, []).gm
        actual = gm(inp)
        self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_decompose_expands_compound_ops(self):
        def f(x):
            return torch.nn.functional.softmax(x, dim=-1)

        gm = trace(f, [torch.randn(4, 8)])
        nodes_before = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        gm = decompose(gm, []).gm
        nodes_after = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        # softmax should decompose into exp, sum, div, etc.
        self.assertGreaterEqual(nodes_after, nodes_before)


class TestFuse(TestCase):
    def test_pointwise_ops_fuse(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].target.n_inputs, 1)

    def test_non_pointwise_not_fused(self):
        def f(x, y):
            return torch.matmul(x, y)

        gm = trace(f, [torch.randn(4, 8), torch.randn(8, 4)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 0)

    def test_fused_kernel_has_correct_ops(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            return b

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        for n in gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.target, FusedKernel):
                op_names = [op.op_name for op in n.target.ops]
                self.assertIn("sin", op_names)
                self.assertIn("cos", op_names)
                break

    def test_fused_kernel_input_shapes_stored(self):
        # Verify that FusedKernel records input_shapes so that
        # _generate_kernel_source can emit broadcast-aware index expressions.
        def f(x, w, s):
            # x: [M, N], w: [N], s: [M, 1]  -- same broadcast pattern as RMSNorm
            return x * w * s

        M, N = 4, 8
        x = torch.randn(M, N)
        w = torch.randn(N)
        s = torch.randn(M, 1)
        gm = trace(f, [x, w, s])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        fused = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, FusedKernel)
        ]
        self.assertEqual(len(fused), 1)
        kernel = fused[0].target
        self.assertIsNotNone(kernel.input_shapes)
        self.assertEqual(len(kernel.input_shapes), kernel.n_inputs)

    @torch.no_grad()
    def test_broadcast_fused_kernel_correct_output(self):
        # The fused kernel must correctly handle inputs that broadcast over
        # the output shape — specifically [N] (row) and [M, 1] (column)
        # vectors multiplied with a 2D tensor [M, N].  This is the same
        # pattern as RMSNorm: out[i,j] = x[i,j] * rsqrt[i] * weight[j].
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.passes.triton import triton_lower
        from torch._torchlite.api import codegen

        def rms_scale(x, weight, rsqrt):
            return x * rsqrt * weight

        M, N = 128, 64
        x = torch.randn(M, N, device="cuda")
        weight = torch.randn(N, device="cuda")
        rsqrt = torch.randn(M, 1, device="cuda")

        gm = trace(rms_scale, [x, weight, rsqrt])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = memory_plan(gm, []).gm
        gm = triton_lower(gm, []).gm
        fn = codegen(gm, inference_codegen=True, example_inputs=[x, weight, rsqrt])

        expected = rms_scale(x, weight, rsqrt)
        actual1 = fn(x, weight, rsqrt)
        actual2 = fn(x, weight, rsqrt)

        self.assertEqual(actual1, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(actual2, expected, atol=1e-4, rtol=1e-4)

    @torch.no_grad()
    def test_broadcast_fused_kernel_correct_output_3d(self):
        # 3D output [B, S, D]: the column-vector [B, S, 1] and the row-vector
        # [D] must use offs // D and offs % D respectively, not plain offs.
        # Before the fix, n_out == 2 guard prevented the broadcast paths from
        # firing, causing the kernel to read garbage memory and produce NaN.
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.passes.triton import triton_lower
        from torch._torchlite.api import codegen

        def rms_scale_3d(x, weight, rsqrt):
            return x * rsqrt * weight

        B, S, D = 2, 16, 32
        x = torch.randn(B, S, D, device="cuda")
        weight = torch.randn(D, device="cuda")
        rsqrt = torch.randn(B, S, 1, device="cuda")

        gm = trace(rms_scale_3d, [x, weight, rsqrt])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = memory_plan(gm, []).gm
        gm = triton_lower(gm, []).gm
        fn = codegen(gm, inference_codegen=True, example_inputs=[x, weight, rsqrt])

        expected = rms_scale_3d(x, weight, rsqrt)
        actual = fn(x, weight, rsqrt)
        self.assertFalse(actual.isnan().any(), "output should not contain NaN")
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)


class TestMatmulEpilogue(TestCase):
    def test_matmul_epilogue_fuses_3d_batch(self):
        # With 3D inputs, PyTorch decomposes linear as:
        #   view([B*S, K]) -> mm([B*S, N]) -> _unsafe_view([B, S, N]) -> epilogue
        # Before the fix, matmul_epilogue stopped at _unsafe_view (not pointwise)
        # and a pointwise epilogue like add or mul was left unfused.
        # Use relu as the epilogue (not a residual add, which is deliberately
        # excluded from matmul epilogue fusion to allow cuBLAS + fused pointwise).
        class Linear3DRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(32, 32, bias=False)

            def forward(self, x):
                return torch.relu(self.proj(x))

        model = Linear3DRelu()
        x = torch.randn(2, 8, 32)

        gm = trace(model, [x])
        gm = decompose(gm, [x]).gm
        gm = normalize(gm, [x]).gm
        gm = matmul_epilogue(gm, [x]).gm

        fused = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, MatmulEpilogueKernel)
        ]
        self.assertGreater(len(fused), 0, "expected at least one MatmulEpilogueKernel")
        kernel = fused[0].target
        self.assertIsNotNone(kernel.out_shape)
        self.assertEqual(kernel.out_shape, [2, 8, 32])

    @torch.no_grad()
    def test_matmul_epilogue_3d_correct_output(self):
        # End-to-end correctness: fused matmul+silu+mul on 3D [B,S,D] input
        # must match eager output without NaN.
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        from torch._torchlite.passes.triton import triton_lower
        from torch._torchlite.api import inference_passes, codegen

        class LlamaFFN(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.norm = nn.RMSNorm(d)
                self.gate_proj = nn.Linear(d, h, bias=False)
                self.up_proj = nn.Linear(d, h, bias=False)
                self.down_proj = nn.Linear(h, d, bias=False)

            def forward(self, x):
                h = self.norm(x)
                return x + self.down_proj(
                    torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
                )

        torch.manual_seed(0)
        model = LlamaFFN(64, 128).cuda().eval()
        x = torch.randn(2, 16, 64, device="cuda")

        expected = model(x)
        gm = trace(model, [x])
        pipeline = inference_passes(gm, [x])
        gm = run_passes(gm, [x], pipeline=pipeline)
        actual = gm(x)

        self.assertFalse(actual.isnan().any(), "output should not contain NaN")
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)


class TestMemoryPlan(TestCase):
    def test_pool_assignment(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            c = torch.exp(b)
            return c

        gm = trace(f, [torch.randn(4, 8)])
        gm = memory_plan(gm, []).gm

        pools = set()
        for node in gm.graph.nodes:
            pool = node.meta.get("memory_pool")
            if pool is not None:
                pools.add(pool)
        self.assertGreater(len(pools), 0)

    def test_stats_computed(self):
        def f(x):
            a = torch.sin(x)
            b = torch.cos(a)
            return b

        gm = trace(f, [torch.randn(4, 8)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]

        self.assertIn("naive_alloc", stats)
        self.assertIn("planned_alloc", stats)
        self.assertIn("num_tensors", stats)
        self.assertIn("num_pools", stats)
        self.assertGreater(stats["num_tensors"], 0)

    def test_empty_graph_stats(self):
        gm = trace(lambda x: x, [torch.randn(4)])
        gm = memory_plan(gm, []).gm
        stats = _graph_meta(gm.graph)["memory_stats"]
        self.assertEqual(stats["naive_alloc"], 0)
        self.assertEqual(stats["planned_alloc"], 0)


class TestFullPipeline(TestCase):
    def test_inference_trace_and_passes(self):
        model = TwoLayerMLP(16, 32, 8)
        inp = torch.randn(4, 16)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        actual = gm(inp)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)

    def test_training_sgd_correctness(self):
        torch.manual_seed(42)
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        x = torch.randn(2, 8)
        target = torch.randn(2, 4)

        expected = train_step(x.clone(), target.clone())

        torch.manual_seed(42)
        model2 = TwoLayerMLP(8, 16, 4)
        train_step2 = TrainStep(model2)
        compiled = compile(train_step2, [x.clone(), target.clone()])
        actual = compiled(x.clone(), target.clone())

        self.assertAlmostEqual(actual.item(), expected.item(), places=4)

    def test_pass_ordering_training(self):
        from torch._torchlite import default_passes

        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]
        gm = trace(train_step, [x.clone() for x in inp])

        passes = default_passes(gm, inp)
        self.assertGreater(len(passes), 0)

        for p in passes:
            result = p(gm, inp)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.gm)
            gm = result.gm


class TestDynamicShapes(TestCase):
    def _inference_pipeline(self):
        return [verify_graph, functionalize, dynamize, memory_plan]

    def test_inference_different_batch_size(self):
        model = TwoLayerMLP(16, 32, 8)
        trace_inp = [torch.randn(4, 16)]
        gm = trace(model, trace_inp)
        gm = run_passes(
            gm, trace_inp,
            pipeline=self._inference_pipeline(),
            dynamic_dims={"x_0": [0]},
        )

        for batch in [1, 4, 8, 16]:
            inp = torch.randn(batch, 16)
            expected = model(inp)
            actual = gm(inp)
            self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_compile_inference_different_batch_size(self):
        def f(x):
            return torch.sin(x) * torch.cos(x) + x

        trace_inp = [torch.randn(4, 8)]
        gm = trace(f, trace_inp)
        gm = run_passes(
            gm, trace_inp,
            pipeline=self._inference_pipeline(),
        )

        for batch in [1, 4, 8, 16]:
            inp = torch.randn(batch, 8)
            expected = f(inp)
            actual = gm(inp)
            self.assertEqual(actual, expected, atol=1e-5, rtol=1e-5)

    def test_training_same_batch_size(self):
        torch.manual_seed(0)
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        trace_x = torch.randn(4, 8)
        trace_target = torch.randn(4, 4)

        compiled = compile(train_step, [trace_x, trace_target])

        x = torch.randn(4, 8)
        target = torch.randn(4, 4)
        loss = compiled(x, target)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_dynamic_dims_explicit_none_defaults_to_batch(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        trace_inp = [torch.randn(4, 8)]
        gm = trace(f, trace_inp)
        gm = run_passes(
            gm, trace_inp,
            pipeline=self._inference_pipeline(),
        )

        out8 = gm(torch.randn(8, 8))
        self.assertEqual(out8.shape, (8, 8))

    def test_empty_dynamic_dims_static(self):
        def f(x):
            return torch.sin(x)

        inp4 = torch.randn(4, 8)
        gm = trace(f, [inp4])
        gm = run_passes(
            gm, [inp4],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        out = gm(inp4)
        self.assertEqual(out.shape, (4, 8))
        expected = torch.sin(inp4)
        self.assertEqual(out, expected, atol=1e-6, rtol=1e-6)


class TestRngFunctionalize(TestCase):
    def test_noop_without_dropout(self):
        def f(x):
            return torch.sin(x) + torch.cos(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = rng_functionalize(gm, []).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIsNot(node.target, _save_rng_state)
                self.assertIsNot(node.target, _load_rng_state)

    def test_noop_without_rng_replay_metadata(self):
        """Dropout in forward but no rng_replay_for metadata on backward
        nodes means dispatcher-based autograd handles the mask internally.
        rng_functionalize should skip inserting save/load nodes."""
        class DropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(8, 4) * 0.01)

            def forward(self, x):
                return torch.nn.functional.dropout(x @ self.w, p=0.5)

        model = DropoutModel()
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm

        gm = rng_functionalize(gm, inp).gm

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertIsNot(node.target, _save_rng_state)
            self.assertIsNot(node.target, _load_rng_state)

    def test_inserts_state_nodes_when_rng_replay_present(self):
        """When backward nodes have rng_replay_for metadata pointing at
        forward dropout nodes, rng_functionalize should insert save_rng_state
        before each forward dropout and load_rng_state before each backward
        replay node."""
        def f(x):
            return torch.nn.functional.dropout(x, p=0.5)

        gm = trace(f, [torch.randn(4, 8)])

        dropout_node = None
        for node in gm.graph.nodes:
            if node.op == "call_function":
                name = getattr(node.target, "__name__", "")
                if name == "dropout":
                    node.meta["phase"] = "forward"
                    dropout_node = node
                    break

        if dropout_node is None:
            self.skipTest("No dropout node found in traced graph")

        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node

        gm.graph.inserting_before(output_node)
        fake_bwd = gm.graph.call_function(torch.neg, (dropout_node,))
        fake_bwd.meta["phase"] = "backward"
        fake_bwd.meta["rng_replay_for"] = dropout_node

        gm.graph.lint()
        gm.recompile()

        gm = rng_functionalize(gm, []).gm

        has_save = any(
            n.op == "call_function" and n.target is _save_rng_state
            for n in gm.graph.nodes
        )
        has_load = any(
            n.op == "call_function" and n.target is _load_rng_state
            for n in gm.graph.nodes
        )
        self.assertTrue(has_save)
        self.assertTrue(has_load)


class TestOptimizer(TestCase):
    def test_sgd_inserts_param_update(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = optimizer(gm, inp, lr=0.01, optimizer_type="sgd").gm

        update_count = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function" and n.target is param_update
        )
        self.assertGreater(update_count, 0)

        has_optimizer_phase = any(
            n.meta.get("phase") == "optimizer"
            for n in gm.graph.nodes
        )
        self.assertTrue(has_optimizer_phase)

    def test_adamw_inserts_adamw_step(self):
        model = TwoLayerMLP(8, 16, 4)
        train_step = TrainStep(model)
        inp = [torch.randn(2, 8), torch.randn(2, 4)]

        gm = trace(train_step, inp)
        gm = functionalize(gm, inp).gm
        gm = autograd_per_op(gm, inp).gm
        gm = save_activations(gm, inp).gm
        gm = optimizer(gm, inp, lr=0.001, optimizer_type="adamw").gm

        adam_count = sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function" and n.target is adamw_step
        )
        self.assertGreater(adam_count, 0)

    def test_noop_without_params(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])
        gm = optimizer(gm, [], lr=0.01).gm

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertIsNot(node.target, param_update)
                self.assertIsNot(node.target, adamw_step)


class TestCudagraphPartition(TestCase):
    def test_all_capturable_single_segment(self):
        def f(x):
            return torch.sin(torch.cos(x)) + x

        gm = trace(f, [torch.randn(4, 8)])
        cudagraph_partition(gm, [])

        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0]["num_nodes"], 0)

    def test_segment_boundary_on_rng_ops(self):
        """Nodes targeting _save_rng_state or _load_rng_state are
        non-capturable and should create segment boundaries."""
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4)])

        output_node = None
        for node in gm.graph.nodes:
            if node.op == "output":
                output_node = node

        gm.graph.inserting_before(output_node)
        rng_node = gm.graph.call_function(_save_rng_state, ())
        rng_node.name = gm.graph._graph_namespace.create_name("rng_state", None)

        sin_node = None
        for node in gm.graph.nodes:
            if node.op == "call_function" and getattr(node.target, "__name__", "") == "sin":
                sin_node = node
                break

        gm.graph.inserting_before(output_node)
        post_node = gm.graph.call_function(torch.cos, (sin_node,))
        post_node.meta["shape"] = sin_node.meta.get("shape", [4])
        post_node.meta["dtype"] = torch.float32

        gm.graph.lint()
        gm.recompile()

        cudagraph_partition(gm, [])
        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertGreaterEqual(len(segments), 2)

    def test_rejects_dynamic_dims(self):
        def f(x):
            return torch.sin(x)

        gm = trace(f, [torch.randn(4, 8)])
        gm = dynamize(gm, [torch.randn(4, 8)]).gm

        has_dynamic = any(
            n.meta.get("dynamic_dims")
            for n in gm.graph.nodes
        )
        if has_dynamic:
            with self.assertRaises(RuntimeError):
                cudagraph_partition(gm, [])

    def test_noop_empty_graph(self):
        gm = trace(lambda x: x, [torch.randn(4)])
        cudagraph_partition(gm, [])
        segments = _graph_meta(gm.graph)["cudagraph_segments"]
        self.assertEqual(len(segments), 0)


class TestTritonCodegen(TestCase):
    def test_generates_triton_code_for_fused_kernels(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("@triton.jit", triton_code)
        self.assertIn("tl.load", triton_code)
        self.assertIn("tl.store", triton_code)

    def test_triton_code_contains_kernel_name(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm

        kernel_name = None
        for n in gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.target, FusedKernel):
                kernel_name = n.target.name
                break

        self.assertIsNotNone(kernel_name)

        gm = triton_codegen(gm, []).gm
        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn(kernel_name, triton_code)

    def test_noop_without_fused_kernels(self):
        def f(x, y):
            return torch.matmul(x, y)

        gm = trace(f, [torch.randn(4, 8), torch.randn(8, 4)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("No fused kernels found", triton_code)

    def test_triton_code_has_sin_cos_ops(self):
        def f(x):
            return torch.sin(torch.cos(x))

        gm = trace(f, [torch.randn(4, 8)])
        gm = decompose(gm, []).gm
        gm = fuse(gm, []).gm
        gm = triton_codegen(gm, []).gm

        triton_code = _graph_meta(gm.graph)["triton_code"]
        self.assertIn("sin", triton_code)
        self.assertIn("cos", triton_code)


class MatmulOnly(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)

    def forward(self, x):
        return x @ self.w


class MatmulChain(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        return (x @ self.w1) @ self.w2


class TestPrecompileSaveLoad(TestCase):
    def _inference_pipeline(self):
        return [verify_graph, functionalize, dynamize, memory_plan]

    def test_matmul_round_trip(self):
        torch.manual_seed(42)
        model = MatmulOnly(16, 8)
        inp = torch.randn(4, 16)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm, [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)
            loaded = precompile_load(artifact_dir, trust_remote_code=True)
            actual = loaded.forward(inp)
            self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_matmul_chain_round_trip(self):
        torch.manual_seed(42)
        model = MatmulChain(8)
        inp = torch.randn(4, 8)
        expected = model(inp)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm, [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)
            loaded = precompile_load(artifact_dir, trust_remote_code=True)
            actual = loaded.forward(inp)
            self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_artifact_directory_structure(self):
        model = MatmulOnly(8, 4)
        inp = torch.randn(2, 8)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm, [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)

            self.assertTrue(
                os.path.exists(os.path.join(artifact_dir, "compiled_module.py"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(artifact_dir, "state_dict.pt"))
            )
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_trust_remote_code_required(self):
        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            with self.assertRaises(ValueError):
                precompile_load(artifact_dir)
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_state_dict_preserved(self):
        torch.manual_seed(42)
        model = MatmulChain(8)
        inp = torch.randn(2, 8)

        gm = trace(model, [inp.clone()])
        gm = run_passes(
            gm, [inp.clone()],
            pipeline=self._inference_pipeline(),
            dynamic_dims={},
        )

        artifact_dir = tempfile.mkdtemp(prefix="torchlite_test_")
        try:
            precompile_save(gm, [inp.clone()], artifact_dir)

            saved_sd = torch.load(
                os.path.join(artifact_dir, "state_dict.pt"),
                weights_only=True,
            )
            for key in model.state_dict():
                self.assertIn(key, saved_sd)
                self.assertEqual(model.state_dict()[key], saved_sd[key])
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)


class TestDynamizeAmbiguousBatch(TestCase):
    def test_dynamize_ambiguous_batch_dim(self):
        """When input is [4, 4] and reshape target is (4, 4), only the
        batch dimension (dim 0) should become dynamic, not dim 1 which
        happens to have the same concrete value."""
        def f(x):
            return x.reshape(4, 4)

        inp = torch.randn(4, 4)
        gm = trace(f, [inp])
        gm = dynamize(gm, [inp], dynamic_dims={"x_0": [0]}).gm

        out8 = gm(torch.randn(8, 4))
        self.assertEqual(out8.shape, (8, 4))


class TestCollectiveFallback(TestCase):
    def test_multi_rank_error_propagates(self):
        """When a real collective raises RuntimeError, it should
        propagate through _try_real_collective rather than being
        silently swallowed."""
        from unittest.mock import patch, MagicMock
        from torch._torchlite.collectives import _try_real_collective

        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("torch.distributed not available")

        mock_funcol = MagicMock()
        mock_funcol.all_reduce.side_effect = RuntimeError("test error")

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.functional_collectives", mock_funcol, create=True), \
             patch.dict("sys.modules", {"torch.distributed.functional_collectives": mock_funcol}):
            with self.assertRaises(RuntimeError):
                _try_real_collective("allreduce", torch.randn(4))


if __name__ == "__main__":
    run_tests()
