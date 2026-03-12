"""Expanded sweep benchmark: torchlite vs torch.compile.

Covers more architectures and dtypes to systematically find regressions:
- SiLU/GELU MLPs (LLM-style gate projections)
- RMSNorm + linear
- LayerNorm + linear
- Multi-head attention
- Transformer blocks (with RMSNorm, SiLU, residual)
- fp16 variants of all the above
- Deeper stacks (ResStack x4, x8)

Run with:
    python torch/_torchlite/examples/bench_sweep2.py
"""

import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._torchlite.api import (
    codegen,
    inference_passes,
    run_passes,
    trace,
)


class SiLUMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GeLUMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class RMSNormLinear(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.linear = nn.Linear(d, h)

    def forward(self, x):
        return self.linear(self.norm(x))


class LayerNormLinear(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear = nn.Linear(d, h)

    def forward(self, x):
        return self.linear(self.norm(x))


class RMSNormSiLUMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)

    def forward(self, x):
        x = self.norm(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ResidualBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class ResStack(nn.Module):
    def __init__(self, d, h, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(d, h) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = SimpleAttention(d, n_heads)
        self.norm2 = nn.RMSNorm(d)
        self.gate = nn.Linear(d, d * ffn_mult)
        self.up = nn.Linear(d, d * ffn_mult)
        self.down = nn.Linear(d * ffn_mult, d)

    def forward(self, x):
        h = self.attn(self.norm1(x))
        x = x + h
        h = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


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


class TanhMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class BottleneckMLP(nn.Module):
    """Wide -> narrow -> wide (autoencoder-style)."""
    def __init__(self, d, bottleneck):
        super().__init__()
        self.enc = nn.Linear(d, bottleneck)
        self.dec = nn.Linear(bottleneck, d)

    def forward(self, x):
        return self.dec(F.relu(self.enc(x)))


class DeepNarrowMLP(nn.Module):
    """Many small layers — stresses Python dispatch overhead."""
    def __init__(self, d, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class LayerNormGeLUMLP(nn.Module):
    """GPT-2 style FFN: LayerNorm -> Linear -> GELU -> Linear."""
    def __init__(self, d, h):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class MultiTransformerBlock(nn.Module):
    """Stack of N transformer blocks."""
    def __init__(self, d, n_heads, n_layers, ffn_mult=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, ffn_mult) for _ in range(n_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class GroupQueryAttention(nn.Module):
    """GQA: fewer KV heads than Q heads (Llama-2 style)."""
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d, n_kv_heads * self.head_dim)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        return self.out(attn)


class SiLUResBlock(nn.Module):
    """Residual block with SiLU instead of ReLU."""
    def __init__(self, d, h):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(x)))


class SiLUResStack(nn.Module):
    def __init__(self, d, h, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([SiLUResBlock(d, h) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
    # === SiLU MLP (LLM gate projections) ===
    ("SiLU MLP d=512", lambda: SiLUMLP(512, 1024), (64, 512), torch.float32),
    ("SiLU MLP d=1024", lambda: SiLUMLP(1024, 2048), (64, 1024), torch.float32),
    ("SiLU MLP d=1024 fp16", lambda: SiLUMLP(1024, 2048), (64, 1024), torch.float16),
    ("SiLU MLP d=1024 bf16", lambda: SiLUMLP(1024, 2048), (64, 1024), torch.bfloat16),
    ("SiLU MLP d=2048", lambda: SiLUMLP(2048, 4096), (64, 2048), torch.float32),
    ("SiLU MLP d=4096 bf16", lambda: SiLUMLP(4096, 11008), (32, 4096), torch.bfloat16),

    # === GeLU MLP ===
    ("GeLU MLP d=512", lambda: GeLUMLP(512, 2048), (64, 512), torch.float32),
    ("GeLU MLP d=1024", lambda: GeLUMLP(1024, 4096), (64, 1024), torch.float32),
    ("GeLU MLP d=1024 fp16", lambda: GeLUMLP(1024, 4096), (64, 1024), torch.float16),
    ("GeLU MLP d=1024 bf16", lambda: GeLUMLP(1024, 4096), (64, 1024), torch.bfloat16),

    # === Tanh MLP ===
    ("Tanh MLP d=512", lambda: TanhMLP(512, 2048), (64, 512), torch.float32),
    ("Tanh MLP d=1024 bf16", lambda: TanhMLP(1024, 4096), (64, 1024), torch.bfloat16),

    # === Norm + Linear ===
    ("RMSNorm+Lin d=512", lambda: RMSNormLinear(512, 1024), (64, 512), torch.float32),
    ("RMSNorm+Lin d=1024", lambda: RMSNormLinear(1024, 2048), (64, 1024), torch.float32),
    ("RMSNorm+Lin d=1024 fp16", lambda: RMSNormLinear(1024, 2048), (64, 1024), torch.float16),
    ("RMSNorm+Lin d=1024 bf16", lambda: RMSNormLinear(1024, 2048), (64, 1024), torch.bfloat16),
    ("LayerNorm+Lin d=512", lambda: LayerNormLinear(512, 1024), (64, 512), torch.float32),
    ("LayerNorm+Lin d=1024", lambda: LayerNormLinear(1024, 2048), (64, 1024), torch.float32),
    ("LayerNorm+Lin d=1024 bf16", lambda: LayerNormLinear(1024, 2048), (64, 1024), torch.bfloat16),

    # === Norm + GeLU MLP (GPT-2 style FFN) ===
    ("LN+GeLU d=512", lambda: LayerNormGeLUMLP(512, 2048), (64, 512), torch.float32),
    ("LN+GeLU d=1024", lambda: LayerNormGeLUMLP(1024, 4096), (64, 1024), torch.float32),
    ("LN+GeLU d=1024 bf16", lambda: LayerNormGeLUMLP(1024, 4096), (64, 1024), torch.bfloat16),

    # === RMSNorm + SiLU MLP (LLM FFN block) ===
    ("RMSNorm+SiLU d=512", lambda: RMSNormSiLUMLP(512, 1024), (64, 512), torch.float32),
    ("RMSNorm+SiLU d=1024", lambda: RMSNormSiLUMLP(1024, 2048), (64, 1024), torch.float32),
    ("RMSNorm+SiLU d=1024 fp16", lambda: RMSNormSiLUMLP(1024, 2048), (64, 1024), torch.float16),
    ("RMSNorm+SiLU d=1024 bf16", lambda: RMSNormSiLUMLP(1024, 2048), (64, 1024), torch.bfloat16),
    ("RMSNorm+SiLU d=4096 bf16", lambda: RMSNormSiLUMLP(4096, 11008), (32, 4096), torch.bfloat16),

    # === Bottleneck MLP (autoencoder-style) ===
    ("Bottleneck 1024->128", lambda: BottleneckMLP(1024, 128), (64, 1024), torch.float32),
    ("Bottleneck 2048->256 bf16", lambda: BottleneckMLP(2048, 256), (64, 2048), torch.bfloat16),

    # === Deep narrow MLP (dispatch overhead stress test) ===
    ("DeepNarrow d=256 x8", lambda: DeepNarrowMLP(256, 8), (64, 256), torch.float32),
    ("DeepNarrow d=512 x8", lambda: DeepNarrowMLP(512, 8), (64, 512), torch.float32),
    ("DeepNarrow d=512 x16", lambda: DeepNarrowMLP(512, 16), (64, 512), torch.float32),

    # === 3-layer MLP ===
    ("3layer d=1024", lambda: ThreeLayerMLP(1024, 2048), (128, 1024), torch.float32),
    ("3layer d=2048", lambda: ThreeLayerMLP(2048, 4096), (128, 2048), torch.float32),
    ("3layer d=2048 bf16", lambda: ThreeLayerMLP(2048, 4096), (128, 2048), torch.bfloat16),

    # === Residual stacks (ReLU) ===
    ("ResStack d=512 x4", lambda: ResStack(512, 1024, 4), (64, 512), torch.float32),
    ("ResStack d=1024 x4", lambda: ResStack(1024, 2048, 4), (64, 1024), torch.float32),
    ("ResStack d=2048 x4", lambda: ResStack(2048, 4096, 4), (64, 2048), torch.float32),
    ("ResStack d=1024 x8", lambda: ResStack(1024, 2048, 8), (64, 1024), torch.float32),
    ("ResStack d=1024 x4 bf16", lambda: ResStack(1024, 2048, 4), (64, 1024), torch.bfloat16),

    # === Residual stacks (SiLU) ===
    ("SiLURes d=512 x4", lambda: SiLUResStack(512, 1024, 4), (64, 512), torch.float32),
    ("SiLURes d=1024 x4", lambda: SiLUResStack(1024, 2048, 4), (64, 1024), torch.float32),
    ("SiLURes d=1024 x4 bf16", lambda: SiLUResStack(1024, 2048, 4), (64, 1024), torch.bfloat16),

    # === Attention ===
    ("Attn d=512 4h", lambda: SimpleAttention(512, 4), (4, 64, 512), torch.float32),
    ("Attn d=512 4h fp16", lambda: SimpleAttention(512, 4), (4, 64, 512), torch.float16),
    ("Attn d=512 8h", lambda: SimpleAttention(512, 8), (4, 64, 512), torch.float32),
    ("Attn d=1024 8h", lambda: SimpleAttention(1024, 8), (4, 64, 1024), torch.float32),
    ("Attn d=1024 8h bf16", lambda: SimpleAttention(1024, 8), (4, 64, 1024), torch.bfloat16),
    ("Attn d=512 seq=256", lambda: SimpleAttention(512, 4), (4, 256, 512), torch.float32),
    ("Attn d=1024 seq=256 bf16", lambda: SimpleAttention(1024, 8), (4, 256, 1024), torch.bfloat16),

    # === Group-query attention (Llama-2 style) ===
    ("GQA d=1024 8q/2kv", lambda: GroupQueryAttention(1024, 8, 2), (4, 64, 1024), torch.float32),
    ("GQA d=1024 8q/2kv bf16", lambda: GroupQueryAttention(1024, 8, 2), (4, 64, 1024), torch.bfloat16),
    ("GQA d=4096 32q/8kv bf16", lambda: GroupQueryAttention(4096, 32, 8), (2, 64, 4096), torch.bfloat16),

    # === Full transformer blocks ===
    ("TFBlock d=512", lambda: TransformerBlock(512, 4), (4, 64, 512), torch.float32),
    ("TFBlock d=512 fp16", lambda: TransformerBlock(512, 4), (4, 64, 512), torch.float16),
    ("TFBlock d=512 bf16", lambda: TransformerBlock(512, 4), (4, 64, 512), torch.bfloat16),
    ("TFBlock d=1024", lambda: TransformerBlock(1024, 8), (4, 64, 1024), torch.float32),
    ("TFBlock d=1024 fp16", lambda: TransformerBlock(1024, 8), (4, 64, 1024), torch.float16),
    ("TFBlock d=1024 bf16", lambda: TransformerBlock(1024, 8), (4, 64, 1024), torch.bfloat16),
    ("TFBlock d=2048 bf16", lambda: TransformerBlock(2048, 16), (2, 64, 2048), torch.bfloat16),

    # === Multi-layer transformer stacks ===
    ("TFStack d=512 x2", lambda: MultiTransformerBlock(512, 4, 2), (4, 64, 512), torch.float32),
    ("TFStack d=512 x4", lambda: MultiTransformerBlock(512, 4, 4), (4, 64, 512), torch.float32),
    ("TFStack d=1024 x2 bf16", lambda: MultiTransformerBlock(1024, 8, 2), (4, 64, 1024), torch.bfloat16),

    # === Longer sequences ===
    ("TFBlock d=512 seq=256", lambda: TransformerBlock(512, 4), (2, 256, 512), torch.float32),
    ("TFBlock d=512 seq=256 bf16", lambda: TransformerBlock(512, 4), (2, 256, 512), torch.bfloat16),
    ("TFBlock d=1024 seq=256 bf16", lambda: TransformerBlock(1024, 8), (2, 256, 1024), torch.bfloat16),

    # === Large batch (compute-bound regime) ===
    ("SiLU MLP d=1024 B=256", lambda: SiLUMLP(1024, 2048), (256, 1024), torch.float32),
    ("SiLU MLP d=1024 B=256 bf16", lambda: SiLUMLP(1024, 2048), (256, 1024), torch.bfloat16),
    ("ResStack d=1024 x4 B=256", lambda: ResStack(1024, 2048, 4), (256, 1024), torch.float32),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    warmup = int(os.environ.get("TORCHLITE_BENCH_WARMUP", "10"))
    iters = int(os.environ.get("TORCHLITE_BENCH_ITERS", "100"))

    print(f"Sweep benchmark v2: warmup={warmup}, iters={iters}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    header = (
        f"{'Model':<34s}  {'Eager (ms)':<12s}  {'TorchLite':<12s}  "
        f"{'torch.compile':<14s}  {'TL/TC ratio':<12s}  {'Notes'}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for name, model_fn, input_shape, dtype in MODELS:
        try:
            torch.manual_seed(42)
            model = model_fn().to(dtype).cuda().eval()
            x = torch.randn(input_shape, device="cuda", dtype=dtype)

            with torch.no_grad():
                eager_ms = _time_fn(model, [x], warmup, iters)

            torch.manual_seed(42)
            model_tl = model_fn().to(dtype).cuda().eval()
            with torch.no_grad():
                gm = trace(model_tl, [x])
                pipeline = inference_passes(gm, [x])
                gm = run_passes(gm, [x], pipeline=pipeline)
                fn_tl = codegen(gm, inference_codegen=True, example_inputs=[x])
                tl_ms = _time_fn(fn_tl, [x], warmup, iters)

            torch.manual_seed(42)
            model_tc = model_fn().to(dtype).cuda().eval()
            tc = torch.compile(model_tc, fullgraph=True)
            with torch.no_grad():
                tc(x)
                tc_ms = _time_fn(tc, [x], warmup, iters)

            ratio = tl_ms / tc_ms if tc_ms > 0 else float("inf")
            notes = ""
            if ratio > 1.15:
                notes = "<-- SLOWER"
            elif ratio < 0.85:
                notes = "<-- FASTER"

            with torch.no_grad():
                out_eager = model(x)
                out_tl = fn_tl(x)
                max_diff = (out_eager - out_tl).abs().max().item()
                if max_diff > 0.05:
                    notes += f" WRONG(diff={max_diff:.4f})"

            print(
                f"{name:<34s}  {eager_ms:<12.3f}  {tl_ms:<12.3f}  "
                f"{tc_ms:<14.3f}  {ratio:<12.2f}  {notes}"
            )
            results.append((name, eager_ms, tl_ms, tc_ms, ratio, notes))

        except Exception as e:
            print(f"{name:<34s}  FAILED: {e}")
            traceback.print_exc()
            results.append((name, 0, 0, 0, 0, f"FAILED: {e}"))

    print()
    print("=" * 80)
    print("SUMMARY: Models where TorchLite is >15% slower than torch.compile:")
    print("=" * 80)
    regressions = [r for r in results if r[4] > 1.15]
    if regressions:
        for name, _, tl_ms, tc_ms, ratio, notes in regressions:
            print(f"  {name:<34s}  TL={tl_ms:.3f}ms  TC={tc_ms:.3f}ms  ratio={ratio:.2f}x  {notes}")
    else:
        print("  None! All models within 15% of torch.compile.")

    print()
    print("SUMMARY: Models with correctness issues:")
    wrong = [r for r in results if "WRONG" in r[5]]
    if wrong:
        for name, _, _, _, _, notes in wrong:
            print(f"  {name:<34s}  {notes}")
    else:
        print("  None! All models correct.")


if __name__ == "__main__":
    main()
