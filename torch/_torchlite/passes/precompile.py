"""Precompile pass: generate standalone Python module from compiled graph."""
import operator
from typing import List

import torch
from torch.fx import GraphModule
from torch.overrides import resolve_name

from torch._torchlite.passes.common import (
    _graph_meta,
    FusedKernel,
    PassResult,
)


def precompile(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    """Generate a standalone Python module from the compiled graph.

    Emits a self-contained Python file with Triton kernels (if any) and a
    CompiledModule class that reproduces the graph's computation. The
    generated code is stored in graph metadata under "precompiled_code".
    """
    triton_code = _graph_meta(gm.graph).get("triton_code", "")
    lines = ["import torch", ""]

    has_torchlite_ops = any(
        n.op == "call_function"
        and "torch._torchlite.ops" in getattr(n.target, "__module__", "")
        for n in gm.graph.nodes
    )
    has_torchlite_collectives = any(
        n.op == "call_function"
        and "torch._torchlite.collectives" in getattr(n.target, "__module__", "")
        for n in gm.graph.nodes
    )
    if has_torchlite_ops:
        lines.append("from torch._torchlite import ops as torchlite_ops")
    if has_torchlite_collectives:
        lines.append("from torch._torchlite import collectives as torchlite_collectives")
    if has_torchlite_ops or has_torchlite_collectives:
        lines.append("")

    has_triton = triton_code.strip() and triton_code.strip() != "# No fused kernels found"
    if has_triton:
        lines += ["import triton", "import triton.language as tl", "", ""]
        lines.append(triton_code.rstrip())
        lines += ["", ""]

    lines.append("class CompiledModule:")
    lines.append("    def __init__(self, state_dict):")
    lines.append("        self.state_dict = state_dict")
    lines.append("")
    lines.append("")
    lines.append("    def __call__(self, *args, **kwargs):")
    lines.append("        return self.forward(*args, **kwargs)")
    lines.append("")

    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node.name)

    sig = ", ".join(["self"] + placeholders)
    lines.append(f"    def forward({sig}):")

    def _fmt(a):
        if isinstance(a, torch.fx.Node):
            return a.name
        return repr(a)

    def _fmt_container(v):
        if isinstance(v, torch.fx.Node):
            return v.name
        if isinstance(v, tuple):
            return "(" + ", ".join(_fmt_container(x) for x in v) + ("," if len(v) == 1 else "") + ")"
        if isinstance(v, list):
            return "[" + ", ".join(_fmt_container(x) for x in v) + "]"
        if isinstance(v, dict):
            items = ", ".join(f"{repr(k)}: {_fmt_container(x)}" for k, x in v.items())
            return "{" + items + "}"
        return repr(v)

    def _fmt_args(node):
        parts = [_fmt_container(a) for a in node.args]
        for k, v in (node.kwargs or {}).items():
            parts.append(f"{k}={_fmt_container(v)}")
        return ", ".join(parts)

    def _emit_target_expr(target):
        if isinstance(target, torch._ops.OpOverload):
            schema = target._schema.name  # e.g. aten::add
            ns, op = schema.split("::", 1)
            overload = target._overloadname or "default"
            return f"torch.ops.{ns}.{op}.{overload}"

        resolved = resolve_name(target)
        if resolved is not None and resolved.startswith("torch.Tensor."):
            method = resolved.split(".")[-1]
            return f"{method}__METHOD__"

        module = getattr(target, "__module__", "")
        fn_name = getattr(target, "__name__", str(target))
        if "torch._torchlite.ops" in module:
            return f"torchlite_ops.{fn_name}"
        if "torch._torchlite.collectives" in module:
            return f"torchlite_collectives.{fn_name}"
        if module and module.startswith("torch"):
            return f"{module}.{fn_name}"
        return f"torch.{fn_name}"

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        elif node.op == "get_attr":
            lines.append(f"        {node.name} = self.state_dict['{node.target}']")
        elif node.op == "call_function":
            target = node.target
            if isinstance(target, FusedKernel):
                input_nodes = [a for a in node.args if isinstance(a, torch.fx.Node)]
                in_args = ", ".join(a.name for a in input_nodes)
                shape = target.shape
                numel = 1
                for s in (shape or []):
                    numel *= s
                device_ref = input_nodes[0].name if input_nodes else None
                dtype = node.meta.get("dtype", torch.float32)
                dtype_str = str(dtype)
                if device_ref:
                    lines.append(
                        f"        {node.name} = torch.empty({shape}, dtype={dtype_str}, device={device_ref}.device)"
                    )
                else:
                    lines.append(f"        {node.name} = torch.empty({shape}, dtype={dtype_str})")
                lines.append(
                    f"        {target.name}[(({numel} + 1023) // 1024,)]"
                    f"({in_args}, {node.name}, {numel})"
                )
            else:
                if target is operator.getitem:
                    lines.append(f"        {node.name} = {_fmt(node.args[0])}[{_fmt(node.args[1])}]")
                    continue
                elif target is torch.Tensor.copy_:
                    lines.append(f"        {_fmt(node.args[0])}.copy_({_fmt(node.args[1])})")
                    continue
                expr = _emit_target_expr(target)
                if expr.endswith("__METHOD__"):
                    method = expr.replace("__METHOD__", "")
                    obj = _fmt_container(node.args[0])
                    arg_parts = [_fmt_container(a) for a in node.args[1:]]
                    arg_parts += [
                        f"{k}={_fmt_container(v)}"
                        for k, v in (node.kwargs or {}).items()
                    ]
                    args_str = ", ".join(arg_parts)
                    lines.append(f"        {node.name} = {obj}.{method}({args_str})")
                else:
                    args_str = _fmt_args(node)
                    lines.append(f"        {node.name} = {expr}({args_str})")
        elif node.op == "output":
            args = node.args[0]
            if isinstance(args, (tuple, list)):
                ret = ", ".join(_fmt(a) for a in args)
                lines.append(f"        return ({ret})")
            else:
                lines.append(f"        return {_fmt(args)}")
    lines.append("")

    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    import sys")
    lines.append("    state_dict = torch.load(sys.argv[1]) if len(sys.argv) > 1 else {}")
    lines.append("    mod = CompiledModule(state_dict)")
    ph_shapes = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ph_shapes[node.name] = node.meta.get("shape", [1])
    in_str = ", ".join(f"torch.randn({ph_shapes.get(p, [1])})" for p in placeholders)
    lines.append(f"    result = mod.forward({in_str})")
    lines.append("    print('Result:', result)")

    code = "\n".join(lines) + "\n"
    _graph_meta(gm.graph)["precompiled_code"] = code
    return PassResult(gm=gm)
