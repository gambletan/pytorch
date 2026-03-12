import difflib
import inspect
import math
import re
import textwrap

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from pygments import highlight as _pygments_highlight
    from pygments.formatters import TerminalTrueColorFormatter
    from pygments.lexers import DiffLexer, PythonLexer
    _HAS_PYGMENTS = True
except ImportError:
    _HAS_PYGMENTS = False

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

    PythonLexer = _Stub
    DiffLexer = _Stub
    TerminalTrueColorFormatter = _Stub

from torch._torchlite import compile, trace
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._torchlite.passes import (
    _CUDAGRAPH_NON_CAPTURABLE,
    _graph_meta,
    _set_phase,
    activation_checkpoint,
    annotate_dtensor,
    autograd_per_op,
    cudagraph_partition,
    decompose,
    dynamize,
    FusedKernel,
    fuse,
    functionalize,
    memory_plan,
    save_activations,
    optimizer,
    PassResult,
    precompile,
    rng_functionalize,
    subclass_unwrap,
    triton_codegen,
)

if _HAS_PYGMENTS:
    PYTHON_FMT = TerminalTrueColorFormatter(style="monokai")
    DIFF_FMT = TerminalTrueColorFormatter(style="monokai")
else:
    PYTHON_FMT = None
    DIFF_FMT = None


def highlight(code, lexer, formatter):
    if _HAS_PYGMENTS:
        return _pygments_highlight(code, lexer, formatter)
    return code

BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{RESET}\n")


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s):
    return _ANSI_RE.sub("", s)


def print_explanation(text):
    print(f"{DIM}{text}{RESET}\n")


def print_diff(before, after, before_label, after_label):
    before = _strip_ansi(before)
    after = _strip_ansi(after)
    diff_text = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=before_label,
            tofile=after_label,
        )
    )
    if diff_text:
        print(f"{DIM}--- diff ---{RESET}")
        print(highlight(diff_text, DiffLexer(), DIFF_FMT))


def _clean_target(node):
    target = node.target

    if isinstance(target, FusedKernel):
        return target.name

    pkt = getattr(target, "overloadpacket", None)
    if pkt is not None:
        ns = getattr(pkt, "__module__", "").rsplit(".", 1)[-1]
        op_name = getattr(pkt, "__name__", str(target))
        return f"{ns}.{op_name}"

    name = getattr(target, "__name__", None)
    qualname = getattr(target, "__qualname__", None) or ""

    if "Tensor" in qualname and "." in qualname:
        return "torch." + qualname.split(".")[-1]

    module = getattr(target, "__module__", "") or ""
    if name:
        short = qualname.split(".")[-1] if qualname else name
        if module.startswith("torch"):
            return f"torch.{short}"
        return short
    return str(target)


def _shape_comment(node):
    shape = node.meta.get("shape")
    pool = node.meta.get("memory_pool")
    spec = node.meta.get("dtensor_spec")
    parts = []
    if shape is not None:
        parts.append(str(list(shape)))
    if spec is not None:
        kind, dim = spec
        if kind == "_Partial":
            parts.append("Partial")
        elif dim is not None:
            parts.append(f"{kind}({dim})")
        else:
            parts.append(kind)
    if pool is not None:
        parts.append(f"pool={pool}")
    if parts:
        return f"  # {', '.join(parts)}"
    return ""


def _format_arg(a):
    if isinstance(a, torch.fx.Node):
        return a.name
    return repr(a)


def _format_args(node):
    parts = [_format_arg(a) for a in node.args]
    parts += [f"{k}={_format_arg(v)}" for k, v in (node.kwargs or {}).items()]
    return ", ".join(parts)


def format_graph(gm):
    lines = []
    indent = "    "
    current_phase = "forward"
    current_bwd_of = None

    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node.name)

    sig = ", ".join(placeholders)
    lines.append(f"def forward({sig}):")

    for node in gm.graph.nodes:
        if node.op == "output":
            lines.append(f"")
            args = node.args[0]
            if isinstance(args, (tuple, list)):
                out_str = ", ".join(_format_arg(a) for a in args)
                lines.append(f"{indent}return ({out_str})")
            else:
                lines.append(f"{indent}return {_format_arg(args)}")
            continue

        phase = node.meta.get("phase", "forward")
        bwd_of = node.meta.get("bwd_of")

        show_header = False
        if phase != current_phase:
            show_header = True
            current_phase = phase
            current_bwd_of = bwd_of
        elif phase == "backward" and bwd_of and bwd_of != current_bwd_of:
            show_header = True
            current_bwd_of = bwd_of

        if show_header:
            if phase == "backward" and bwd_of:
                label = f"backward ({bwd_of})"
            elif phase == "save":
                label = "saved for backward"
            elif phase == "recompute":
                label = "recompute"
            else:
                label = phase
            lines.append("")
            lines.append(
                f"{indent}# ── {label} "
                f"{'─' * (40 - len(label))}"
            )

        if node.op == "placeholder":
            continue
        elif node.op == "get_attr":
            shape = _shape_comment(node)
            lines.append(f"{indent}{node.name} = self.{node.target}{shape}")
        elif node.op == "call_function":
            target_str = _clean_target(node)
            args_str = _format_args(node)
            shape = _shape_comment(node)
            lines.append(
                f"{indent}{node.name} = {target_str}({args_str}){shape}"
            )

    return "\n".join(lines) + "\n"


def print_graph(gm):
    code = format_graph(gm)
    highlighted = highlight(code, PythonLexer(), PYTHON_FMT)
    out_lines = []
    for line in highlighted.splitlines():
        plain = _strip_ansi(line).strip()
        if plain.startswith("# ──"):
            out_lines.append(f"    {YELLOW}{plain}{RESET}")
        else:
            out_lines.append(line)
    print("\n".join(out_lines))


# ── Model ─────────────────────────────────────────────────


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 32
        n_heads = 2
        head_dim = 16

        self.w1 = nn.Parameter(torch.randn(10, hidden))
        self.b1 = nn.Parameter(torch.randn(hidden))
        self.rms_norm = nn.RMSNorm(hidden)
        self.wq = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wk = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wv = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wo = nn.Parameter(torch.randn(n_heads * head_dim, hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, 10))
        self.b2 = nn.Parameter(torch.randn(10))

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        n_heads, head_dim = 2, 16

        h = x @ self.w1
        h.add_(self.b1)
        h = self.rms_norm(h)
        h = torch.sin(h)

        q = (h @ self.wq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        k = (h @ self.wk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        v = (h @ self.wv).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, n_heads * head_dim)

        h = out @ self.wo
        out = h @ self.w2
        out.add_(self.b2)
        out = torch.cos(out)
        out = torch.dropout(out, 0.5, True)
        return out


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        return ((out - target) ** 2).mean()


# ── Fake Process Group ────────────────────────────────────
store = FakeStore()
dist.init_process_group("fake", rank=0, world_size=2, store=store)
mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))

# ── Source ─────────────────────────────────────────────────
print_header("Source")
print_explanation("  Model (forward pass):")
source = textwrap.dedent(inspect.getsource(MyModel.forward))
print(highlight(source, PythonLexer(), PYTHON_FMT))
print_explanation("  Training step (forward + MSE loss):")
source = textwrap.dedent(inspect.getsource(TrainStep.forward))
print(highlight(source, PythonLexer(), PYTHON_FMT))
print_explanation("  Optimizer (SGD):")
print(highlight("for p in model.parameters():\n    p -= lr * p.grad\n", PythonLexer(), PYTHON_FMT))

model = MyModel()

# Distribute parameters as DTensors for tensor parallelism.
# wo is column-parallel (Shard(1)) and w2 is row-parallel (Shard(0))
# forming a Megatron-style column→row pair with allreduce between them.
# Attention weights and rms_norm stay Replicate since DTensor's batched
# matmul dispatch doesn't support sharded "head" dimensions.
model.w1 = nn.Parameter(distribute_tensor(model.w1, mesh, [Replicate()]))
model.b1 = nn.Parameter(distribute_tensor(model.b1, mesh, [Replicate()]))
model.rms_norm.weight = nn.Parameter(distribute_tensor(model.rms_norm.weight, mesh, [Replicate()]))
model.wq = nn.Parameter(distribute_tensor(model.wq, mesh, [Replicate()]))
model.wk = nn.Parameter(distribute_tensor(model.wk, mesh, [Replicate()]))
model.wv = nn.Parameter(distribute_tensor(model.wv, mesh, [Replicate()]))
model.wo = nn.Parameter(distribute_tensor(model.wo, mesh, [Shard(1)]))
model.w2 = nn.Parameter(distribute_tensor(model.w2, mesh, [Shard(0)]))
model.b2 = nn.Parameter(distribute_tensor(model.b2, mesh, [Replicate()]))

train_step = TrainStep(model)
x = distribute_tensor(torch.randn(2, 4, 10), mesh, [Replicate()])
target = distribute_tensor(torch.randn(2, 4, 10), mesh, [Replicate()])

print_header("DTensor Setup")
print_explanation(
    "  Simulate 2-rank tensor parallelism using a fake process group.\n"
    "  No torchrun or multi-process setup needed."
)
print(f"  {DIM}Mesh: {mesh}{RESET}")
for name, param in model.named_parameters():
    dt = param if isinstance(param, DTensor) else param.data
    if isinstance(dt, DTensor):
        print(
            f"  {DIM}{name}: global {list(dt.shape)}, "
            f"placement {dt.placements}, "
            f"local {list(dt.to_local().shape)}{RESET}"
        )
    else:
        print(f"  {DIM}{name}: {list(param.shape)} (not distributed){RESET}")
print()

# ── Step 1: Trace ─────────────────────────────────────────
gm = trace(train_step, [x, target])
print_header("Step 1: Trace")
print_explanation(
    "  Trace the forward pass: execute the model once and record every\n"
    "  torch operation into a graph of simple function calls."
)
prev = format_graph(gm)
print_graph(gm)

# ── Step 2: Functionalize ─────────────────────────────────
result = functionalize(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 2: Functionalize")
print_explanation(
    "  Remove in-place mutations: replace ops like add_() with their\n"
    "  functional equivalents (add) so the graph is side-effect free."
)
print_graph(gm)

print_diff(prev, cur, "trace", "functionalize")
prev = cur

# ── Step 3: Dynamize ─────────────────────────────────────
prev_dyn = cur
result = dynamize(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 3: Dynamize")
print_explanation(
    "  Insert explicit size-extraction nodes for dynamic dimensions\n"
    "  (batch dim) and replace concrete shape literals in reshape/view.\n"
    "  Enables running with different batch sizes without retracing."
)
print_graph(gm)

print_diff(prev, cur, "functionalize", "dynamize")
prev = cur

# ── Step 4: Annotate DTensor ─────────────────────────────
result = annotate_dtensor(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 4: Annotate DTensor")
print_explanation(
    "  Stamp dtensor_spec metadata on get_attr and placeholder nodes\n"
    "  for DTensor placement info (Shard/Replicate). This seeds the\n"
    "  placement annotations that subclass_unwrap reads."
)
print_graph(gm)

print_diff(prev, cur, "dynamize", "annotate_dtensor")
prev = cur

# ── Step 5: Subclass Unwrap ──────────────────────────────
pre_shapes = {}
pre_node_names = set()
for node in gm.graph.nodes:
    pre_node_names.add(node.name)
    shape = node.meta.get("shape")
    if shape is not None:
        pre_shapes[node.name] = list(shape)

result = subclass_unwrap(gm, [x, target], world_size=2)
gm = result.gm
cur = format_graph(gm)
print_header("Step 5: Subclass Unwrap")
print_explanation(
    "  Unwrap tensor subclasses: the model uses DTensor for tensor\n"
    "  parallelism. This pass lowers DTensor operations to local tensor\n"
    "  ops + explicit collectives (allgather, allreduce), converting\n"
    "  global shapes to per-rank local shapes."
)

collectives = []
for node in gm.graph.nodes:
    if node.name not in pre_node_names and node.op == "call_function":
        target_name = getattr(node.target, "__name__", str(node.target))
        arg_name = (
            node.args[0].name
            if node.args and isinstance(node.args[0], torch.fx.Node)
            else "?"
        )
        collectives.append((target_name, arg_name))

if collectives:
    print(f"  {DIM}Collectives inserted:{RESET}")
    for target_name, arg_name in collectives:
        print(f"    {BOLD}{target_name}{RESET}{DIM}({arg_name}){RESET}")
    print()

changed = []
for node in gm.graph.nodes:
    if node.name not in pre_shapes:
        continue
    shape = node.meta.get("shape")
    spec = node.meta.get("dtensor_spec")
    if shape is not None and pre_shapes[node.name] != list(shape):
        spec_str = ""
        if spec:
            kind, dim = spec
            spec_str = f"  {kind}({dim})" if dim is not None else f"  {kind}"
        changed.append((node.name, pre_shapes[node.name], list(shape), spec_str))

if changed:
    print(f"  {DIM}Shapes (global → local, world_size=2):{RESET}")
    for name, old, new, spec_str in changed:
        print(f"    {name}: {old} → {new}{spec_str}")
    print()

print_graph(gm)

print_diff(prev, cur, "annotate_dtensor", "subclass_unwrap")
prev = cur

# ── Step 6: Autograd Per-Op ───────────────────────────────
result = autograd_per_op(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 6: Autograd Per-Op")
print_explanation(
    "  Differentiate the graph per-op: walk forward ops in reverse and\n"
    "  emit backward nodes for each op using known derivative rules.\n"
    "  No monolithic make_fx retrace — backward is built explicitly."
)
print_graph(gm)

print_diff(prev, cur, "subclass_unwrap", "autograd_per_op")
prev = cur

# ── Step 7: RNG State Functionalize ──────────────────────
result = rng_functionalize(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 7: RNG State Functionalize")
print_explanation(
    "  Functionalize RNG state: dropout is non-deterministic, so backward\n"
    "  needs to replay the exact same random mask. This pass inserts\n"
    "  save_rng_state before forward dropout and load_rng_state before\n"
    "  the backward dropout to ensure deterministic gradient computation."
)
print_graph(gm)

print_diff(prev, cur, "autograd_per_op", "rng_functionalize")
prev = cur

# ── Step 8: Save Activations ──────────
result = save_activations(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 8: Save Activations")
print_explanation(
    "  Analyze the joint forward+backward graph to find the optimal cut:\n"
    "  which forward activations to save for backward, vs. which to\n"
    "  recompute. Insert explicit save_for_backward nodes at the\n"
    "  forward/backward boundary."
)
num_saves = sum(
    1 for n in gm.graph.nodes
    if n.op == "call_function"
    and getattr(n.target, "__name__", "") == "save_for_backward"
)
print(f"  {DIM}Activations saved for backward: {num_saves}{RESET}\n")
print_graph(gm)

print_diff(prev, cur, "rng_functionalize", "save_activations")
prev = cur

# ── Step 9: Activation Checkpointing ────────────────────
num_saves_before = num_saves
result = activation_checkpoint(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 9: Activation Checkpointing")
print_explanation(
    "  Reduce peak memory by recomputing cheap ops (sin, cos, add, sub)\n"
    "  during backward instead of saving their outputs. Replaces\n"
    "  save_for_backward nodes with recomputation nodes."
)
num_saves_after = sum(
    1 for n in gm.graph.nodes
    if n.op == "call_function"
    and getattr(n.target, "__name__", "") == "save_for_backward"
)
num_recomputed = sum(
    1 for n in gm.graph.nodes
    if n.meta.get("phase") == "recompute"
)
print(f"  {DIM}Saved: {num_saves_before} → {num_saves_after}, recomputed: {num_recomputed}{RESET}\n")
print_graph(gm)

print_diff(prev, cur, "save_activations", "activation_checkpoint")
prev = cur

# ── Step 10: Optimizer ─────────────────────────────────────
result = optimizer(gm, [x, target], lr=0.01)
gm = result.gm
cur = format_graph(gm)
print_header("Step 10: Optimizer")
print_explanation(
    "  Fuse the optimizer into the graph: for each parameter,\n"
    "  compute w_new = w - lr * grad_w and update in-place.\n"
    "  Gradients are consumed; the graph returns only the loss."
)
print_graph(gm)

print_diff(prev, cur, "activation_checkpoint", "optimizer")
prev = cur

# ── Step 11: Memory Planning ─────────────────────────────
result = memory_plan(gm, [x, target])
gm = result.gm
cur = format_graph(gm)
print_header("Step 11: Memory Planning")
print_explanation(
    "  Analyze tensor lifetimes and assign memory pools. Tensors whose\n"
    "  lifetimes don't overlap share a pool, reducing total allocations\n"
    "  and eliminating fragmentation."
)
stats = _graph_meta(gm.graph)["memory_stats"]
naive_kb = stats["naive_alloc"] / 1024
planned_kb = stats["planned_alloc"] / 1024
savings_pct = (
    (1 - stats["planned_alloc"] / stats["naive_alloc"]) * 100
    if stats["naive_alloc"] > 0
    else 0
)
print(f"  {DIM}Tensors: {stats['num_tensors']}, Pools: {stats['num_pools']}{RESET}")
print(f"  {DIM}Total allocation (naive):   {naive_kb:.1f} KB ({stats['num_tensors']} buffers){RESET}")
print(f"  {DIM}Total allocation (planned): {planned_kb:.1f} KB ({stats['num_pools']} buffers){RESET}")
print(f"  {DIM}Savings: {savings_pct:.0f}%{RESET}")

pool_members = {}
for node in gm.graph.nodes:
    pid = node.meta.get("memory_pool")
    if pid is not None:
        pool_members.setdefault(pid, []).append(node.name)

shared = {pid: members for pid, members in pool_members.items() if len(members) > 1}
if shared:
    print(f"\n  {DIM}Shared pools:{RESET}")
    for pid in sorted(shared):
        names = shared[pid]
        print(f"    {BOLD}pool {pid}{RESET}: {', '.join(names)}")
print()
print_graph(gm)

print_diff(prev, cur, "optimizer", "memory_plan")
prev = cur

# ── Step 12: Decompose ────────────────────────────────────
# decompose is now a proper graph pass — it walks nodes individually and
# inlines per-op decompositions. No whole-graph make_fx retrace needed,
# so it works with custom markers (allgather, save_rng_state, etc.).
prev = format_graph(gm)
result = decompose(gm, [x, target])
decomposed_gm = result.gm
cur = format_graph(decomposed_gm)
print_header("Step 12: Decompose")
print_explanation(
    "  Lower to ATen core ops: replace high-level operations (matmul,\n"
    "  mean, etc.) with their primitive implementations."
)
print_graph(decomposed_gm)

print_diff(prev, cur, "memory_plan", "decompose")
prev = cur

# ── Step 13: Fusion ──────────────────────────────────────
prev = cur
result = fuse(decomposed_gm, [x, target])
decomposed_gm = result.gm
cur = format_graph(decomposed_gm)
print_header("Step 13: Fusion")
print_explanation(
    "  Identify groups of compatible elementwise ops that can be fused\n"
    "  into a single GPU kernel, reducing memory bandwidth overhead.\n"
    "  Each group is replaced by a single fused placeholder node."
)
fused_count = 0
for node in decomposed_gm.graph.nodes:
    if node.op == "call_function" and isinstance(node.target, FusedKernel):
        kernel = node.target
        fused_count += 1
        shape_str = str(kernel.shape) if kernel.shape else "?"
        print(f"  {BOLD}{kernel.name}{RESET} {DIM}({shape_str}){RESET}")
        ops_str = " → ".join(op.op_name for op in kernel.ops)
        print(f"    {DIM}Ops:{RESET}    {YELLOW}{ops_str}{RESET}")
        inputs_str = ", ".join(
            a.name for a in node.args if isinstance(a, torch.fx.Node)
        )
        print(f"    {DIM}Inputs:{RESET} {inputs_str}")
        print()
if fused_count == 0:
    print(f"  {DIM}No multi-op fusion opportunities found.{RESET}\n")

print_graph(decomposed_gm)
print_diff(prev, cur, "decompose", "fuse")

# ── Step 14: Triton Code Generation ──────────────────────
result = triton_codegen(decomposed_gm, [x, target])
decomposed_gm = result.gm
triton_code = _graph_meta(decomposed_gm.graph)["triton_code"]
print_header("Step 14: Triton Code Generation")
print_explanation(
    "  Generate Triton GPU kernels for each fusion group. Each kernel\n"
    "  loads inputs, computes the fused ops, and stores the result\n"
    "  in a single pass over memory."
)
print_explanation("  FX graph (entire program with fused kernels):")
print_graph(decomposed_gm)
print_explanation("\n  Generated Triton kernels:")
print(highlight(triton_code, PythonLexer(), PYTHON_FMT))

# ── Step 15: Precompile ──────────────────────────────────
result = precompile(decomposed_gm, [x, target])
decomposed_gm = result.gm
precompile_code = _graph_meta(decomposed_gm.graph)["precompiled_code"]
print_header("Step 15: Precompile")
print_explanation(
    "  Generate a standalone Python file with pre-compiled Triton\n"
    "  kernels and a CompiledModule class. Can be loaded and executed\n"
    "  without the original model or torch.compile."
)
print(highlight(precompile_code, PythonLexer(), PYTHON_FMT))

# ── Step 16: CUDA Graph Partition ────────────────────────
# The main pipeline graph (gm) went through dynamize and has dynamic_dims,
# which CUDA graphs reject (static shapes required). We demonstrate two
# things: (a) the pass correctly rejects dynamic-dim graphs, and (b) on
# a static-shape graph (traced fresh without dynamize), the pass runs the
# full segmentation analysis.
print_header("Step 16: CUDA Graph Partition")
print_explanation(
    "  Analyze the graph for CUDA-graph compatibility. Classify each op\n"
    "  as capturable or not, and group contiguous capturable ops into\n"
    "  numbered segments. Non-capturable ops (CPU RNG state, autograd\n"
    "  fallbacks) act as segment boundaries."
)

print(f"  {BOLD}Dynamic-dim graph:{RESET}")
try:
    cudagraph_partition(gm, [x, target])
except RuntimeError as e:
    print(f"  {DIM}Correctly rejected: {e}{RESET}\n")

# Trace a fresh static-shape graph with the full pass pipeline (minus
# dynamize) so we can show the partition analysis on a representative
# training graph.
from torch._torchlite import trace as _trace
static_gm = _trace(train_step, [x, target])
static_gm = functionalize(static_gm, [x, target]).gm
static_gm = autograd_per_op(static_gm, [x, target]).gm
static_gm = rng_functionalize(static_gm, [x, target]).gm

result = cudagraph_partition(static_gm, [x, target])
partition_gm = result.gm
segments = _graph_meta(partition_gm.graph)["cudagraph_segments"]

non_capturable = []
for node in partition_gm.graph.nodes:
    if node.op != "call_function":
        continue
    name = getattr(node.target, "__name__", "")
    if name in _CUDAGRAPH_NON_CAPTURABLE:
        non_capturable.append(node.name)

print(f"  {BOLD}Static-shape graph:{RESET}")
print(f"  {DIM}Segments: {len(segments)}{RESET}")
for sid, info in sorted(segments.items()):
    print(
        f"    {BOLD}segment {sid}{RESET}: "
        f"{info['start']} → {info['end']} "
        f"({info['num_nodes']} nodes)"
    )
if non_capturable:
    print(f"\n  {DIM}Non-capturable nodes (segment boundaries):{RESET}")
    for name in non_capturable:
        print(f"    {YELLOW}{name}{RESET}")
else:
    print(f"\n  {DIM}All nodes are capturable (whole-graph capture possible){RESET}")
print()

# ── Runtime Demo ──────────────────────────────────────────
# The runtime demo uses plain tensors because the fake process group
# doesn't perform actual collective operations needed for DTensor execution.
print_header("Runtime: Training Step")
model2 = MyModel()
train_step2 = TrainStep(model2)
x_rt = torch.randn(2, 4, 10)
target_rt = torch.randn(2, 4, 10)

w1_before = model2.w1.data.clone()
w2_before = model2.w2.data.clone()

compiled_fn = compile(train_step2, [x_rt, target_rt], lr=0.01)

loss = compiled_fn(x_rt, target_rt)
print(f"  loss: {loss.item():.6f}")
print(f"  w1 changed: {not torch.equal(model2.w1.data, w1_before)}")
print(f"  w2 changed: {not torch.equal(model2.w2.data, w2_before)}")
print(f"  w1 delta norm: {(model2.w1.data - w1_before).norm().item():.6f}")
print(f"  w2 delta norm: {(model2.w2.data - w2_before).norm().item():.6f}")

dist.destroy_process_group()
