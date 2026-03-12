"""Entry points for the torchlite compiler.

The compiler has three phases:
  trace()      - capture a model into an FX graph
  run_passes() - run all graph transformation passes
  codegen()    - convert the transformed graph into a callable
compile() = trace() + run_passes() + codegen().
"""
import functools
import importlib.util
import operator
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule
from torch.overrides import TorchFunctionMode

from torch._torchlite.collectives import _has_dtensor_params
from torch._torchlite.passes import (
    _create_name,
    _deep_getattr,
    _graph_meta,
    activation_checkpoint,
    annotate_dtensor,
    autograd_per_op,
    cudagraph_partition,
    decompose,
    dynamize,
    fuse,
    fuse_add_layer_norm,
    fuse_add_rms_norm,
    functionalize,
    matmul_epilogue,
    memory_plan,
    normalize,
    sdpa_pattern,
    simplify_views,
    optimizer,
    precompile,
    rng_functionalize,
    save_activations,
    subclass_unwrap,
    triton_codegen,
    triton_lower,
    verify_graph,
)


_SKIP_FUNC_NAMES = frozenset({"__get__", "__set__", "__delete__"})

# Mapping from Tensor property descriptors to their torch.* functional
# equivalents. When TorchFunctionMode intercepts a __get__ call for one
# of these properties, we record the functional version in the FX graph
# instead of the un-serializable method-wrapper.
_DESCRIPTOR_TO_FUNC = {}
for _attr, _func in [
    ("real", torch.real),
    ("imag", torch.imag),
    ("T", torch.t),
    ("mT", lambda x: torch.transpose(x, -2, -1)),
]:
    _desc = getattr(torch.Tensor, _attr, None)
    if _desc is not None:
        _DESCRIPTOR_TO_FUNC[_desc] = _func
del _attr, _func, _desc
_triton_cache_lock = threading.Lock()


def _is_tensor(v):
    return isinstance(v, torch.Tensor)


def _is_dtensor_like(v) -> bool:
    return hasattr(v, "placements") and hasattr(v, "device_mesh")


def _graph_has_dtensor_params(
    gm: GraphModule,
    example_inputs: Optional[List[torch.Tensor]] = None,
) -> bool:
    # Fast path: metadata already stamped by annotate_dtensor.
    if _has_dtensor_params(gm):
        return True

    # Before annotate_dtensor has run, detect DTensor values directly from
    # module attributes referenced by get_attr nodes.
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            val = _deep_getattr(gm, node.target)
        except AttributeError:
            continue
        if _is_dtensor_like(val):
            return True

    if example_inputs is not None:
        for inp in example_inputs:
            if _is_dtensor_like(inp):
                return True

    return False


def _track_tensor(tensor, node, tensor_to_node, _tracked_tensors):
    tensor_to_node[id(tensor)] = node
    _tracked_tensors.append(tensor)
    # DTensors wrap an inner _local_tensor. Track it too so that
    # operations inside to_local() (which access the inner tensor)
    # remain linked to this graph node.
    inner = getattr(tensor, "_local_tensor", None)
    if inner is not None and id(inner) not in tensor_to_node:
        tensor_to_node[id(inner)] = node
        _tracked_tensors.append(inner)


def trace(
    model: Union[nn.Module, Callable],
    example_inputs: List[torch.Tensor],
) -> GraphModule:
    """Capture a model's forward pass into an FX graph.

    Executes the model once with a TorchFunctionMode that intercepts every
    torch operation and records it as a node in an FX graph. Parameters
    become get_attr nodes, inputs become placeholder nodes, and operations
    become call_function nodes with their torch.* callable as the target.
    """
    graph = Graph()
    tensor_to_node = {}
    # Prevent GC from collecting tensors whose id() we use as keys in
    # tensor_to_node. Without this, Python could reuse a dead object's id
    # for a new tensor, causing silent mis-mapping to the wrong FX node.
    _tracked_tensors = []
    # Untracked tensors (e.g. block masks from create_block_mask, -inf
    # constants) that appear as arguments to traced ops. Stored as
    # GraphModule attributes so FX can reference them by name.
    _const_tensors = {}

    if isinstance(model, nn.Module):
        for name, param in model.named_parameters():
            node = graph.get_attr(name)
            node.meta["shape"] = list(param.shape)
            node.meta["dtype"] = param.dtype
            _track_tensor(param, node, tensor_to_node, _tracked_tensors)

    placeholders = []
    for i, inp in enumerate(example_inputs):
        node = graph.placeholder(f"x_{i}")
        if _is_tensor(inp):
            node.meta["shape"] = list(inp.shape)
            node.meta["dtype"] = inp.dtype
            _track_tensor(inp, node, tensor_to_node, _tracked_tensors)
        elif isinstance(inp, (list, tuple)):
            for j, item in enumerate(inp):
                if _is_tensor(item):
                    get_node = graph.call_function(
                        operator.getitem, (node, j)
                    )
                    get_node.meta["shape"] = list(item.shape)
                    get_node.meta["dtype"] = item.dtype
                    _track_tensor(
                        item, get_node, tensor_to_node, _tracked_tensors
                    )
        placeholders.append(node)

    class Tracer(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            # When torch.compile/dynamo is actively tracing (e.g. for
            # flex_attention or create_block_mask compiled inside the model),
            # torchlite must be transparent — otherwise dynamo tries to trace
            # into our tracing logic and hits unsupported builtins.
            if torch.compiler.is_compiling():
                return func(*args, **kwargs)

            # Higher-order ops (e.g. flex_attention) contain non-serializable
            # arguments (functions, dicts) that FX cannot codegen. Instead of
            # recording the higher-order op itself, call the underlying eager
            # implementation directly (bypassing HOP dispatch) with this mode
            # re-pushed so the internal computation (matmul, softmax, …) is
            # captured as individual graph nodes.
            if isinstance(func, torch._ops.HigherOrderOperator):
                from torch._C import DispatchKey
                impl = func.py_kernels.get(DispatchKey.CompositeExplicitAutograd)
                if impl is not None:
                    with self:
                        result = impl(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result

            # Execute the real op with the mode temporarily popped
            result = func(*args, **kwargs)

            func_name = getattr(func, "__name__", None)
            if func_name in _SKIP_FUNC_NAMES:
                if func_name == "__get__" and _is_tensor(result) and args and _is_tensor(args[0]):
                    descriptor = getattr(func, "__self__", None)
                    torch_func = _DESCRIPTOR_TO_FUNC.get(descriptor)
                    if torch_func is not None:
                        func = torch_func
                        args = (args[0],)
                    else:
                        return result
                else:
                    return result

            # Build proxy args: replace tensors with their graph nodes,
            # leave scalars/ints/floats as-is.
            # For DTensors created by from_local() wrapping a tracked local
            # tensor, look up the inner tensor's node.
            def _lookup(v):
                if id(v) in tensor_to_node:
                    return tensor_to_node[id(v)]
                inner = getattr(v, "_local_tensor", None)
                if inner is not None and id(inner) in tensor_to_node:
                    return tensor_to_node[id(inner)]
                return None

            # Check whether any argument is an already-tracked tensor
            # BEFORE creating constant nodes for untracked tensors.
            # Without this ordering, create_block_mask's internal ops
            # (which operate on purely untracked vmap'd tensors) would
            # be pulled into the graph and fail at replay time.
            #
            # We distinguish two cases where no args are tracked:
            #   a) Args contain untracked tensors (e.g. create_block_mask
            #      internals operating on vmap'd tensors) -> skip the op.
            #   b) Args contain NO tensors at all (factory ops like
            #      torch.zeros, torch.empty, torch.arange) -> record it.
            def _any_tracked(vals):
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        if _any_tracked(v):
                            return True
                    elif isinstance(v, dict):
                        if _any_tracked(v.values()):
                            return True
                    elif _is_tensor(v) and _lookup(v) is not None:
                        return True
                return False

            def _has_any_tensor(vals):
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        if _has_any_tensor(v):
                            return True
                    elif isinstance(v, dict):
                        if _has_any_tensor(v.values()):
                            return True
                    elif _is_tensor(v):
                        return True
                return False

            has_tracked = (
                _any_tracked(args) or _any_tracked(kwargs.values())
            )
            if not has_tracked:
                has_untracked_tensors = (
                    _has_any_tensor(args) or _has_any_tensor(kwargs.values())
                )
                if has_untracked_tensors:
                    return result

            def to_proxy(v):
                if _is_tensor(v):
                    node_for_v = _lookup(v)
                    if node_for_v is not None:
                        return node_for_v
                    const_name = f"_const_{len(_const_tensors)}"
                    _const_tensors[const_name] = v
                    const_node = graph.get_attr(const_name)
                    const_node.meta["shape"] = list(v.shape)
                    const_node.meta["dtype"] = v.dtype
                    tensor_to_node[id(v)] = const_node
                    _tracked_tensors.append(v)
                    return const_node
                if isinstance(v, tuple):
                    return tuple(to_proxy(x) for x in v)
                if isinstance(v, list):
                    return [to_proxy(x) for x in v]
                if isinstance(v, dict):
                    return {k: to_proxy(x) for k, x in v.items()}
                return v

            proxy_args = tuple(to_proxy(a) for a in args)
            proxy_kwargs = {k: to_proxy(v) for k, v in kwargs.items()}

            name = getattr(func, "__name__", str(func))
            node = graph.call_function(func, proxy_args, proxy_kwargs)
            node.name = _create_name(graph, name)

            def _track_result(result, parent_node):
                if _is_tensor(result):
                    parent_node.meta["shape"] = list(result.shape)
                    parent_node.meta["dtype"] = result.dtype
                    _track_tensor(
                        result, parent_node,
                        tensor_to_node, _tracked_tensors,
                    )
                elif isinstance(result, (tuple, list)):
                    for i, r in enumerate(result):
                        get_node = graph.call_function(
                            operator.getitem, (parent_node, i)
                        )
                        _track_result(r, get_node)

            _track_result(result, node)

            return result

    # Disable inner torch.compile calls (e.g. flex_attention compiled
    # inside the model) so their operations flow through our
    # TorchFunctionMode and become part of the traced graph.
    with torch.compiler.set_stance("force_eager"), Tracer():
        output = model(*example_inputs)

    def _resolve_output(o):
        if _is_tensor(o):
            if id(o) in tensor_to_node:
                return tensor_to_node[id(o)]
            inner = getattr(o, "_local_tensor", None)
            if inner is not None and id(inner) in tensor_to_node:
                return tensor_to_node[id(inner)]
            raise RuntimeError(f"Output tensor not tracked: {o.shape}")
        if isinstance(o, (tuple, list)):
            return type(o)(_resolve_output(x) for x in o)
        return o

    out_nodes = _resolve_output(output)

    graph.output(out_nodes)
    graph.lint()

    root = model if isinstance(model, nn.Module) else nn.Module()
    for const_name, const_val in _const_tensors.items():
        root.register_buffer(const_name, const_val, persistent=False)
    gm = GraphModule(root, graph)
    gm = normalize(gm, example_inputs).gm
    return gm


def default_passes(
    gm: Optional[GraphModule] = None,
    example_inputs: Optional[List[torch.Tensor]] = None,
    *,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
    world_size: int = 1,
) -> List[Callable]:
    """Return the default pass pipeline.

    Each entry is a callable with signature (gm, example_inputs) -> PassResult.
    When gm is provided, DTensor-specific passes (annotate_dtensor,
    subclass_unwrap) are only included if the graph contains DTensor
    parameters or DTensor example inputs. When gm is None, distributed
    passes are always included for backward compatibility.
    """
    include_distributed = (
        gm is None
        or _graph_has_dtensor_params(gm, example_inputs)
    )

    passes = [
        verify_graph,
        functionalize,
        functools.partial(dynamize, dynamic_dims=dynamic_dims),
    ]
    if include_distributed:
        passes.append(annotate_dtensor)
        passes.append(functools.partial(subclass_unwrap, world_size=world_size))
    passes += [
        autograd_per_op,
        rng_functionalize,
        save_activations,
        activation_checkpoint,
        functools.partial(optimizer, lr=lr, optimizer_type=optimizer_type),
        memory_plan,
    ]
    return passes


def inference_passes(
    gm: Optional[GraphModule] = None,
    example_inputs: Optional[List[torch.Tensor]] = None,
    *,
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
    world_size: int = 1,
) -> List[Callable]:
    """Return the inference-only pass pipeline.

    Like default_passes but without autograd, activation checkpointing,
    or optimizer passes. Suitable for inference-only compilation.
    """
    include_distributed = (
        gm is None
        or _graph_has_dtensor_params(gm, example_inputs)
    )

    passes = [
        verify_graph,
        functionalize,
        functools.partial(dynamize, dynamic_dims=dynamic_dims),
    ]
    if include_distributed:
        passes.append(annotate_dtensor)
        passes.append(functools.partial(subclass_unwrap, world_size=world_size))
    passes += [
        decompose,
        simplify_views,
        sdpa_pattern,
        matmul_epilogue,
        fuse,
        fuse_add_rms_norm,
        triton_lower,
        memory_plan,
    ]
    return passes


def run_passes(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    pipeline: Optional[List[Callable]] = None,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
    world_size: int = 1,
) -> GraphModule:
    """Run graph transformation passes on a traced graph.

    This is the middle step of the trace → run_passes → codegen pipeline.
    When pipeline is provided, runs exactly those passes in order.
    Otherwise runs the default pipeline. DTensor-specific passes are
    auto-included only when the graph contains DTensor parameters.
    """
    if pipeline is None:
        pipeline = default_passes(
            gm,
            example_inputs=example_inputs,
            lr=lr, optimizer_type=optimizer_type,
            dynamic_dims=dynamic_dims, world_size=world_size,
        )

    for p in pipeline:
        gm = p(gm, example_inputs).gm

    return gm


def timed_run_passes(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    *,
    pipeline: Optional[List[Callable]] = None,
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
    world_size: int = 1,
) -> tuple:
    """Run passes and return (gm, timings_dict).

    Like run_passes but records wall-clock time for each pass.
    Returns a tuple of (graph_module, dict mapping pass name to seconds).
    """
    import time

    if pipeline is None:
        pipeline = default_passes(
            gm,
            example_inputs=example_inputs,
            lr=lr, optimizer_type=optimizer_type,
            dynamic_dims=dynamic_dims, world_size=world_size,
        )

    timings: Dict[str, float] = {}
    for p in pipeline:
        name = getattr(p, "__name__", None)
        if name is None:
            inner = getattr(p, "func", p)
            name = getattr(inner, "__name__", repr(p))
        t0 = time.perf_counter()
        gm = p(gm, example_inputs).gm
        timings[name] = time.perf_counter() - t0

    return gm, timings


def codegen_inference(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable:
    """Generate an optimized forward function from the post-triton_lower graph.

    Walks the FX graph and emits Python source for a forward() that:
    - Pre-allocates intermediate buffers and reuses them via memory_pool metadata
    - Calls Triton kernels directly (bypassing _TritonKernelModule.__call__)
    - Uses out= variants for mm/addmm to write into pre-allocated buffers
    - Clones the final output to avoid aliasing pre-allocated memory

    The generated code is compiled via exec() into a closure that captures
    pre-allocated buffers, Triton kernel functions, and model parameters.
    """
    from torch._torchlite.passes.triton import (
        _AddLayerNormModule,
        _AddRmsNormModule,
        _TritonKernelModule,
        _TritonMatmulModule,
    )

    graph = gm.graph
    lines = []
    namespace: Dict[str, object] = {"torch": torch}

    # Collect pool info: for each pool_id, track the max numel and
    # whether all nodes sharing the pool have the same shape.
    # When shapes are uniform we can pre-allocate a shaped buffer
    # and use out= directly. When shapes differ we skip out= for
    # mismatched nodes to avoid resize overhead.
    pool_info: Dict[int, dict] = {}
    for node in graph.nodes:
        pool_id = node.meta.get("memory_pool")
        if pool_id is None:
            continue
        shape = node.meta.get("shape")
        dtype = node.meta.get("dtype", torch.float32)
        if shape is None:
            continue
        shape_t = tuple(shape)
        numel = 1
        for s in shape:
            numel *= s
        if pool_id not in pool_info:
            pool_info[pool_id] = {
                "max_numel": numel,
                "shape": shape_t,
                "dtype": dtype,
                "uniform": True,
            }
        else:
            existing = pool_info[pool_id]
            if numel > existing["max_numel"]:
                existing["max_numel"] = numel
            if shape_t != existing["shape"] or dtype != existing["dtype"]:
                existing["uniform"] = False

    device = example_inputs[0].device if example_inputs else torch.device("cuda")
    for pool_id, info in pool_info.items():
        if info["uniform"]:
            buf = torch.empty(
                list(info["shape"]), dtype=info["dtype"], device=device
            )
        else:
            buf = torch.empty(
                info["max_numel"], dtype=info["dtype"], device=device
            )
        buf_name = f"_buf{pool_id}"
        namespace[buf_name] = buf

    # Pre-compute views for non-uniform pool nodes. Each node with a
    # non-uniform pool needs a view of the flat buffer at its shape.
    # Pre-computing avoids aten::slice + aten::view on every call.
    pool_views: Dict[str, str] = {}
    for node in graph.nodes:
        pool_id = node.meta.get("memory_pool")
        if pool_id is None or pool_info.get(pool_id, {}).get("uniform", True):
            continue
        shape = node.meta.get("shape")
        if shape is None:
            continue
        numel = 1
        for s in shape:
            numel *= s
        view_name = f"_pv_{node.name}"
        buf = namespace[f"_buf{pool_id}"]
        namespace[view_name] = buf[:numel].view(list(shape))
        pool_views[node.name] = view_name

    # Register parameters/attrs and triton modules in namespace
    for node in graph.nodes:
        if node.op == "get_attr":
            attr_val = _deep_getattr(gm, node.target)
            safe_name = node.name
            namespace[safe_name] = attr_val
        elif node.op == "call_module":
            mod = _deep_getattr(gm, node.target)
            if isinstance(mod, _TritonKernelModule):
                fn_name = f"_triton_fn_{node.name}"
                namespace[fn_name] = mod.triton_fn
                namespace[f"_triton_numel_{node.name}"] = mod.numel
                namespace[f"_triton_grid_{node.name}"] = (
                    (mod.numel + 1023) // 1024,
                )
            elif isinstance(mod, _TritonMatmulModule):
                mod_ref = f"_matmul_mod_{node.name}"
                namespace[mod_ref] = mod
            elif isinstance(mod, _AddRmsNormModule):
                mod_ref = f"_add_rms_mod_{node.name}"
                namespace[mod_ref] = mod
            elif isinstance(mod, _AddLayerNormModule):
                mod_ref = f"_add_ln_mod_{node.name}"
                namespace[mod_ref] = mod

    # Force benchmark on _TritonMatmulModule instances during codegen
    # setup so _use_cublas is populated before we generate code.
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        mod = _deep_getattr(gm, node.target)
        if isinstance(mod, _TritonMatmulModule) and mod._use_cublas is None:
            dummy_input = torch.empty(
                (mod.M, mod.K), dtype=mod.dtype, device=device
            )
            dummy_weight = torch.empty(
                (mod.K, mod.N), dtype=mod.dtype, device=device
            )
            dummy_bias = (
                torch.empty(mod.N, dtype=mod.dtype, device=device)
                if mod.has_bias else None
            )
            dummy_extras = [
                torch.empty(s, dtype=mod.dtype, device=device)
                for s in mod.extra_shapes
            ]
            mod._benchmark_backends(
                dummy_input, dummy_weight, dummy_bias, dummy_extras
            )

    # Map node -> variable name in generated code
    node_to_var: Dict[str, str] = {}

    # Build placeholder signature
    placeholder_names = []
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholder_names.append(node.name)
            node_to_var[node.name] = node.name

    lines.append(f"def forward({', '.join(placeholder_names)}):")

    # Map get_attr nodes
    for node in graph.nodes:
        if node.op == "get_attr":
            node_to_var[node.name] = node.name

    # Generate code for each op node
    output_var = None
    out_node = None
    _handled_nodes: set = set()
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.name in _handled_nodes:
            continue

        if node.op == "output":
            out_arg = node.args[0]
            if isinstance(out_arg, torch.fx.Node):
                output_var = node_to_var.get(out_arg.name, out_arg.name)
                out_node = out_arg
            else:
                output_var = str(out_arg)
            continue

        pool_id = node.meta.get("memory_pool")
        var_name = node.name

        def _arg_str(arg):
            if isinstance(arg, torch.fx.Node):
                return node_to_var.get(arg.name, arg.name)
            if isinstance(arg, (torch.device, torch.dtype, torch.layout, torch.memory_format)):
                const_name = f"_const_{id(arg)}"
                namespace[const_name] = arg
                return const_name
            return repr(arg)

        if node.op == "call_module":
            mod = _deep_getattr(gm, node.target)
            if isinstance(mod, _TritonKernelModule):
                # Triton pointwise kernels assume contiguous layout for
                # broadcast index expressions.  Force .contiguous() on each
                # input to avoid wrong results from sliced/permuted views.
                input_args = ", ".join(
                    f"{_arg_str(a)}.contiguous()" for a in node.args
                )
                fn_name = f"_triton_fn_{node.name}"
                grid_name = f"_triton_grid_{node.name}"
                numel_name = f"_triton_numel_{node.name}"
                pool_uniform = (
                    pool_id is not None
                    and pool_info[pool_id]["uniform"]
                )
                if pool_id is not None and pool_uniform:
                    buf_name = f"_buf{pool_id}"
                    node_to_var[node.name] = buf_name
                    lines.append(
                        f"    {fn_name}[{grid_name}]"
                        f"({input_args}, {buf_name}, {numel_name})"
                    )
                elif pool_id is not None:
                    pv = pool_views.get(node.name)
                    if pv:
                        node_to_var[node.name] = pv
                        lines.append(
                            f"    {fn_name}[{grid_name}]"
                            f"({input_args}, {pv}, {numel_name})"
                        )
                    else:
                        view_var = f"_v_{var_name}"
                        buf_name = f"_buf{pool_id}"
                        shape_repr = repr(mod.shape)
                        lines.append(
                            f"    {view_var} = {buf_name}[:{mod.numel}]"
                            f".view({shape_repr})"
                        )
                        node_to_var[node.name] = view_var
                        lines.append(
                            f"    {fn_name}[{grid_name}]"
                            f"({input_args}, {view_var}, {numel_name})"
                        )
                else:
                    node_to_var[node.name] = var_name
                    shape_repr = repr(mod.shape)
                    dtype_name = f"_dtype_{var_name}"
                    namespace[dtype_name] = mod.dtype
                    lines.append(
                        f"    {var_name} = torch.empty("
                        f"{shape_repr}, dtype={dtype_name}, "
                        f"device={_arg_str(node.args[0])}.device)"
                    )
                    lines.append(
                        f"    {fn_name}[{grid_name}]"
                        f"({input_args}, {var_name}, {numel_name})"
                    )
            elif isinstance(mod, _TritonMatmulModule):
                mod_ref = f"_matmul_mod_{node.name}"
                input_args = ", ".join(_arg_str(a) for a in node.args)
                pool_uniform = (
                    pool_id is not None
                    and pool_info[pool_id]["uniform"]
                )
                can_inline = mod._use_cublas
                if can_inline:
                    # Inline the cuBLAS matmul + epilogue directly into the
                    # generated code. This avoids the Python method dispatch
                    # overhead of _forward_into_buf (~30us per call).
                    if pool_id is not None and pool_uniform:
                        buf_name = f"_buf{pool_id}"
                    elif pool_id is not None:
                        mn = mod.M * mod.N
                        view_var = f"_v_{var_name}"
                        buf_name = view_var
                        lines.append(
                            f"    {view_var} = _buf{pool_id}[:{mn}]"
                            f".view({mod.M}, {mod.N})"
                        )
                    else:
                        buf_name = f"_buf_unpool_{node.name}"
                        namespace[buf_name] = torch.empty(
                            (mod.M, mod.N), dtype=mod.dtype, device=device,
                        )
                    node_to_var[node.name] = var_name

                    input_var = _arg_str(node.args[0])
                    weight_var = _arg_str(node.args[1])

                    if mod.has_bias:
                        extra_args = node.args[3:]
                    else:
                        extra_args = node.args[2:]

                    # When there are no epilogue ops and bias is present,
                    # use addmm.out (1 kernel) instead of mm.out + add_
                    # (2 kernels). Otherwise emit mm.out as the matmul step.
                    use_addmm = (
                        mod.has_bias and not mod.epilogue_ops
                    )
                    if use_addmm:
                        mm_op = f"_addmm_op_{node.name}"
                        namespace[mm_op] = torch.ops.aten.addmm.out
                        bias_var = _arg_str(node.args[2])
                        lines.append(
                            f"    {mm_op}({bias_var}, {input_var}, "
                            f"{weight_var}, out={buf_name})"
                        )
                    else:
                        mm_op = f"_mm_op_{node.name}"
                        namespace[mm_op] = torch.ops.aten.mm.out
                        lines.append(
                            f"    {mm_op}({input_var}, {weight_var}, "
                            f"out={buf_name})"
                        )

                    if mod._can_fuse_epilogue and mod.epilogue_ops:
                        # Use a single Triton pointwise kernel for
                        # bias + epilogue + extras instead of separate
                        # eager ops. This saves 1-3 kernel launches.
                        # The epilogue writes in-place back to the accumulator
                        # buffer (out_ptr == acc_ptr), which is safe because
                        # each thread reads acc[i] then writes out[i] with no
                        # cross-element dependencies.
                        mod._ensure_epilogue_fn()
                        ep_fn_ref = f"_ep_fn_{node.name}"
                        namespace[ep_fn_ref] = mod._epilogue_fn

                        acc_var = buf_name
                        if (mod.out_shape is not None
                                and list(mod.out_shape) != [mod.M, mod.N]):
                            acc_var = f"_acc_{node.name}"
                            lines.append(
                                f"    {acc_var} = {buf_name}.view("
                                f"{repr(mod.out_shape)})"
                            )

                        numel = mod.M * mod.N
                        grid_val = (numel + 1023) // 1024
                        ep_grid = f"_ep_grid_{node.name}"
                        namespace[ep_grid] = (grid_val,)

                        # Build kernel args: acc, [bias,] [extras...], out, N, numel
                        # out == acc (in-place) to avoid a torch.empty allocation
                        kernel_args = [acc_var]
                        if mod.has_bias:
                            kernel_args.append(_arg_str(node.args[2]))
                        for ea in extra_args:
                            evar = _arg_str(ea)
                            flat_ref = f"_flat_{node.name}_{evar}"
                            namespace[flat_ref] = None  # placeholder
                            lines.append(
                                f"    {flat_ref} = {evar}.reshape(-1) "
                                f"if {evar}.ndim > 1 else {evar}"
                            )
                            kernel_args.append(flat_ref)
                        kernel_args.extend([acc_var, str(mod.N), str(numel)])
                        args_str = ", ".join(kernel_args)
                        lines.append(
                            f"    {ep_fn_ref}[{ep_grid}]({args_str})"
                        )
                        lines.append(f"    {var_name} = {acc_var}")
                    elif mod.epilogue_ops:
                        if mod.has_bias:
                            bias_var = _arg_str(node.args[2])
                            lines.append(
                                f"    {buf_name}.add_({bias_var})"
                            )

                        acc_var = buf_name
                        if (mod.out_shape is not None
                                and list(mod.out_shape) != [mod.M, mod.N]):
                            acc_var = f"_acc_{node.name}"
                            lines.append(
                                f"    {acc_var} = {buf_name}.view("
                                f"{repr(mod.out_shape)})"
                            )

                        from torch._torchlite.passes.triton import (
                            _TORCH_OP_MAP,
                            _TORCH_UNARY_INPLACE_OP_MAP,
                        )
                        for op in mod.epilogue_ops:
                            if (op.op_name in _TORCH_UNARY_INPLACE_OP_MAP
                                    and len(op.args) == 1
                                    and op.args[0][0] in ("acc", "tmp")):
                                fn_ref = f"_ep_{node.name}_{op.op_name}"
                                namespace[fn_ref] = _TORCH_UNARY_INPLACE_OP_MAP[op.op_name]
                                lines.append(f"    {fn_ref}({acc_var})")
                            elif op.op_name in ("add", "mul") and len(op.args) == 2:
                                def _ep_arg(arg):
                                    tag = arg[0]
                                    if tag in ("acc", "tmp"):
                                        return acc_var
                                    elif tag == "input":
                                        return _arg_str(node.args[arg[1]])
                                    elif tag == "extra":
                                        return _arg_str(extra_args[arg[1]])
                                    elif tag == "const":
                                        return repr(arg[1])
                                a_str = _ep_arg(op.args[0])
                                b_str = _ep_arg(op.args[1])
                                inplace_method = "add_" if op.op_name == "add" else "mul_"
                                py_op = "+" if op.op_name == "add" else "*"
                                if a_str == acc_var or b_str == acc_var:
                                    other = b_str if a_str == acc_var else a_str
                                    lines.append(
                                        f"    {acc_var}.{inplace_method}({other})"
                                    )
                                else:
                                    acc_var = var_name
                                    lines.append(
                                        f"    {var_name} = {a_str} {py_op} {b_str}"
                                    )
                            else:
                                from torch._torchlite.passes.triton import _apply_epilogue_inplace
                                ep_fn = f"_ep_fn_{node.name}"
                                namespace[ep_fn] = _apply_epilogue_inplace
                                namespace[f"_ep_ops_{node.name}"] = mod.epilogue_ops
                                extras_str = ", ".join(
                                    _arg_str(a) for a in extra_args
                                )
                                bias_str = (
                                    _arg_str(node.args[2]) if mod.has_bias
                                    else "None"
                                )
                                acc_var = var_name
                                lines.append(
                                    f"    {var_name} = {ep_fn}({buf_name}, "
                                    f"_ep_ops_{node.name}, {input_var}, "
                                    f"{weight_var}, {bias_str}, "
                                    f"[{extras_str}])"
                                )
                                break

                        if acc_var != var_name:
                            lines.append(f"    {var_name} = {acc_var}")
                    else:
                        # No epilogue ops — addmm.out already handled bias
                        # above if present, nothing extra needed here.
                        if (mod.out_shape is not None
                                and list(mod.out_shape) != [mod.M, mod.N]):
                            lines.append(
                                f"    {var_name} = {buf_name}.view("
                                f"{repr(mod.out_shape)})"
                            )
                        else:
                            lines.append(f"    {var_name} = {buf_name}")
                elif pool_id is not None and pool_uniform:
                    buf_name = f"_buf{pool_id}"
                    node_to_var[node.name] = var_name
                    lines.append(
                        f"    {var_name} = {mod_ref}._forward_into_buf("
                        f"{buf_name}, {input_args})"
                    )
                elif pool_id is not None:
                    mn = mod.M * mod.N
                    view_var = f"_v_{var_name}"
                    buf_name = f"_buf{pool_id}"
                    node_to_var[node.name] = var_name
                    lines.append(
                        f"    {view_var} = {buf_name}[:{mn}]"
                        f".view({mod.M}, {mod.N})"
                    )
                    lines.append(
                        f"    {var_name} = {mod_ref}._forward_into_buf("
                        f"{view_var}, {input_args})"
                    )
                else:
                    node_to_var[node.name] = var_name
                    lines.append(
                        f"    {var_name} = {mod_ref}({input_args})"
                    )
            elif isinstance(mod, _AddRmsNormModule):
                mod_ref = f"_add_rms_mod_{node.name}"
                input_args = ", ".join(_arg_str(a) for a in node.args)
                node_to_var[node.name] = var_name

                if not mod.has_add:
                    lines.append(
                        f"    {var_name} = {mod_ref}({input_args})"
                    )
                else:
                    getitem_nodes = {}
                    for user in node.users:
                        if (
                            user.op == "call_function"
                            and user.target is operator.getitem
                            and len(user.args) >= 2
                        ):
                            getitem_nodes[user.args[1]] = user

                    add_gi = getitem_nodes.get(0)
                    norm_gi = getitem_nodes.get(1)

                    def _pool_buf_for(gi_node):
                        if gi_node is None:
                            return None
                        pid = gi_node.meta.get("memory_pool")
                        if pid is None:
                            return None
                        pi = pool_info.get(pid)
                        if pi is None:
                            return None
                        if pi["uniform"]:
                            return f"_buf{pid}"
                        pv = pool_views.get(gi_node.name)
                        return pv

                    add_buf = _pool_buf_for(add_gi)
                    norm_buf = _pool_buf_for(norm_gi)

                    if add_buf and norm_buf:
                        lines.append(
                            f"    {mod_ref}.forward_into_bufs("
                            f"{input_args}, {add_buf}, {norm_buf})"
                        )
                        if add_gi:
                            node_to_var[add_gi.name] = add_buf
                            _handled_nodes.add(add_gi.name)
                        if norm_gi:
                            node_to_var[norm_gi.name] = norm_buf
                            _handled_nodes.add(norm_gi.name)
                    elif add_buf and not norm_buf:
                        norm_var = f"_norm_{node.name}"
                        dtype_ref = f"_dtype_{node.name}"
                        namespace[dtype_ref] = mod.dtype
                        shape_repr = repr(mod.shape)
                        lines.append(
                            f"    {norm_var} = torch.empty("
                            f"{shape_repr}, dtype={dtype_ref}, "
                            f"device={_arg_str(node.args[0])}.device)"
                        )
                        lines.append(
                            f"    {mod_ref}.forward_into_bufs("
                            f"{input_args}, {add_buf}, {norm_var})"
                        )
                        if add_gi:
                            node_to_var[add_gi.name] = add_buf
                            _handled_nodes.add(add_gi.name)
                        if norm_gi:
                            node_to_var[norm_gi.name] = norm_var
                            _handled_nodes.add(norm_gi.name)
                    else:
                        lines.append(
                            f"    {var_name} = {mod_ref}({input_args})"
                        )
            elif isinstance(mod, _AddLayerNormModule):
                mod_ref = f"_add_ln_mod_{node.name}"
                input_args = ", ".join(_arg_str(a) for a in node.args)
                node_to_var[node.name] = var_name

                if not mod.has_add:
                    lines.append(
                        f"    {var_name} = {mod_ref}({input_args})"
                    )
                else:
                    getitem_nodes = {}
                    for user in node.users:
                        if (
                            user.op == "call_function"
                            and user.target is operator.getitem
                            and len(user.args) >= 2
                        ):
                            getitem_nodes[user.args[1]] = user

                    add_gi = getitem_nodes.get(0)
                    norm_gi = getitem_nodes.get(1)

                    def _pool_buf_for_ln(gi_node):
                        if gi_node is None:
                            return None
                        pid = gi_node.meta.get("memory_pool")
                        if pid is None:
                            return None
                        pi = pool_info.get(pid)
                        if pi is None:
                            return None
                        if pi["uniform"]:
                            return f"_buf{pid}"
                        pv = pool_views.get(gi_node.name)
                        return pv

                    add_buf = _pool_buf_for_ln(add_gi)
                    norm_buf = _pool_buf_for_ln(norm_gi)

                    if add_buf and norm_buf:
                        lines.append(
                            f"    {mod_ref}.forward_into_bufs("
                            f"{input_args}, {add_buf}, {norm_buf})"
                        )
                        if add_gi:
                            node_to_var[add_gi.name] = add_buf
                            _handled_nodes.add(add_gi.name)
                        if norm_gi:
                            node_to_var[norm_gi.name] = norm_buf
                            _handled_nodes.add(norm_gi.name)
                    else:
                        lines.append(
                            f"    {var_name} = {mod_ref}({input_args})"
                        )
            else:
                node_to_var[node.name] = var_name
                input_args = ", ".join(_arg_str(a) for a in node.args)
                lines.append(f"    {var_name} = {node.target}({input_args})")
            continue

        if node.op == "call_function":
            target = node.target
            args_str = ", ".join(_arg_str(a) for a in node.args)
            kwargs_parts = []
            for k, v in node.kwargs.items():
                kwargs_parts.append(f"{k}={_arg_str(v)}")

            target_name = getattr(target, "__name__", str(target))
            is_mm = target in (
                torch.ops.aten.mm.default,
                torch.ops.aten.addmm.default,
            )
            is_view_like = target in (
                torch.ops.aten.t.default,
                torch.ops.aten.transpose.int,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.permute.default,
                torch.ops.aten._unsafe_view.default,
            )

            pool_uniform = (
                pool_id is not None
                and pool_info[pool_id]["uniform"]
            )
            node_dtype = node.meta.get("dtype", torch.float32)
            decompose_addmm = (
                target == torch.ops.aten.addmm.default
                and node_dtype == torch.float32
            )
            if is_mm and pool_id is not None and (
                target == torch.ops.aten.mm.default or decompose_addmm
            ):
                ns_target = f"_op_{node.name}"
                namespace[ns_target] = torch.ops.aten.mm.out
                pv = pool_views.get(node.name)
                if pool_uniform:
                    buf_name = f"_buf{pool_id}"
                    node_to_var[node.name] = buf_name
                elif pv:
                    buf_name = pv
                    node_to_var[node.name] = pv
                else:
                    shape = node.meta.get("shape", [])
                    numel = 1
                    for s in shape:
                        numel *= s
                    view_var = f"_v_{var_name}"
                    buf_name = view_var
                    shape_repr = repr(list(shape))
                    lines.append(
                        f"    {view_var} = _buf{pool_id}[:{numel}]"
                        f".view({shape_repr})"
                    )
                    node_to_var[node.name] = view_var

                if target == torch.ops.aten.mm.default:
                    lines.append(
                        f"    {ns_target}({args_str}, out={buf_name})"
                    )
                elif target == torch.ops.aten.addmm.default:
                    bias_arg = _arg_str(node.args[0])
                    input_arg = _arg_str(node.args[1])
                    weight_arg = _arg_str(node.args[2])
                    lines.append(
                        f"    {ns_target}({input_arg}, {weight_arg}, out={buf_name})"
                    )
                    lines.append(
                        f"    {buf_name}.add_({bias_arg})"
                    )
            elif is_mm and pool_id is not None:
                ns_target = f"_op_{node.name}"
                namespace[ns_target] = torch.ops.aten.addmm.out
                pv = pool_views.get(node.name)
                if pool_uniform:
                    buf_name = f"_buf{pool_id}"
                    node_to_var[node.name] = buf_name
                elif pv:
                    buf_name = pv
                    node_to_var[node.name] = pv
                else:
                    shape = node.meta.get("shape", [])
                    numel = 1
                    for s in shape:
                        numel *= s
                    view_var = f"_v_{var_name}"
                    buf_name = view_var
                    shape_repr = repr(list(shape))
                    lines.append(
                        f"    {view_var} = _buf{pool_id}[:{numel}]"
                        f".view({shape_repr})"
                    )
                    node_to_var[node.name] = view_var
                lines.append(
                    f"    {ns_target}({args_str}, out={buf_name})"
                )
            elif decompose_addmm and pool_id is None:
                node_to_var[node.name] = var_name
                ns_target = f"_op_{node.name}"
                namespace[ns_target] = torch.mm
                bias_arg = _arg_str(node.args[0])
                input_arg = _arg_str(node.args[1])
                weight_arg = _arg_str(node.args[2])
                lines.append(
                    f"    {var_name} = {ns_target}({input_arg}, {weight_arg})"
                )
                lines.append(
                    f"    {var_name}.add_({bias_arg})"
                )
            elif is_view_like:
                node_to_var[node.name] = var_name
                self_arg = _arg_str(node.args[0])

                def _shape_args(shape_arg):
                    """Convert a shape argument (list or varargs) to a
                    comma-separated string of resolved variable names.
                    Handles FX Nodes inside lists for dynamic shapes.
                    """
                    if isinstance(shape_arg, (list, tuple)):
                        return ", ".join(_arg_str(s) for s in shape_arg)
                    return _arg_str(shape_arg)

                if target == torch.ops.aten.t.default:
                    input_node = node.args[0]
                    if input_node.op == "get_attr":
                        # Pre-compute the transposed view at codegen time
                        # so we don't pay ~0.6us per .t() call every forward.
                        namespace[var_name] = namespace[input_node.name].t()
                        node_to_var[node.name] = var_name
                    else:
                        lines.append(f"    {var_name} = {self_arg}.t()")
                elif target == torch.ops.aten.transpose.int:
                    rest = ", ".join(_arg_str(a) for a in node.args[1:])
                    lines.append(
                        f"    {var_name} = {self_arg}.transpose({rest})"
                    )
                elif target in (
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                ):
                    shape_str = _shape_args(node.args[1])
                    lines.append(
                        f"    {var_name} = {self_arg}.reshape({shape_str})"
                    )
                elif target == torch.ops.aten._unsafe_view.default:
                    shape_str = _shape_args(node.args[1])
                    lines.append(
                        f"    {var_name} = {self_arg}.view({shape_str})"
                    )
                elif target == torch.ops.aten.expand.default:
                    shape_str = _shape_args(node.args[1])
                    lines.append(
                        f"    {var_name} = {self_arg}.expand({shape_str})"
                    )
                elif target == torch.ops.aten.permute.default:
                    perm_str = _shape_args(node.args[1])
                    lines.append(
                        f"    {var_name} = {self_arg}.permute({perm_str})"
                    )
                elif target == torch.ops.aten.slice.Tensor:
                    rest = ", ".join(_arg_str(a) for a in node.args[1:])
                    lines.append(
                        f"    {var_name} = {self_arg}[{rest}]"
                    )
                else:
                    ns_target = f"_op_{node.name}"
                    namespace[ns_target] = target
                    all_args = args_str
                    if kwargs_parts:
                        all_args += ", " + ", ".join(kwargs_parts)
                    lines.append(
                        f"    {var_name} = {ns_target}({all_args})"
                    )
            elif pool_id is None:
                node_to_var[node.name] = var_name
                if target is operator.getitem:
                    lines.append(
                        f"    {var_name} = {_arg_str(node.args[0])}"
                        f"[{_arg_str(node.args[1])}]"
                    )
                elif target == torch.ops.aten.unbind.int:
                    self_arg = _arg_str(node.args[0])
                    dim_arg = _arg_str(node.args[1]) if len(node.args) > 1 else "0"
                    lines.append(
                        f"    {var_name} = {self_arg}.unbind({dim_arg})"
                    )
                elif target in (
                    torch.ops.aten.sym_size.int,
                    torch.ops.aten.sym_size.default,
                ):
                    self_arg = _arg_str(node.args[0])
                    rest = ", ".join(_arg_str(a) for a in node.args[1:])
                    lines.append(
                        f"    {var_name} = {self_arg}.size({rest})"
                    )
                elif target == torch.ops.aten.clone.default:
                    self_arg = _arg_str(node.args[0])
                    lines.append(f"    {var_name} = {self_arg}.clone()")
                elif target == torch.ops.aten.select.int:
                    self_arg = _arg_str(node.args[0])
                    rest = ", ".join(_arg_str(a) for a in node.args[1:])
                    lines.append(
                        f"    {var_name} = {self_arg}.select({rest})"
                    )
                else:
                    ns_target = f"_op_{node.name}"
                    namespace[ns_target] = target
                    all_args = args_str
                    if kwargs_parts:
                        all_args += ", " + ", ".join(kwargs_parts)
                    lines.append(
                        f"    {var_name} = {ns_target}({all_args})"
                    )
            else:
                # Generic op with pool — can't use out=, just let it allocate
                node_to_var[node.name] = var_name
                ns_target = f"_op_{node.name}"
                namespace[ns_target] = target
                all_args = args_str
                if kwargs_parts:
                    all_args += ", " + ", ".join(kwargs_parts)
                lines.append(f"    {var_name} = {ns_target}({all_args})")
            continue

    if output_var is not None:
        needs_clone = out_node is not None and out_node.meta.get("memory_pool") is not None
        if needs_clone:
            lines.append(f"    return {output_var}.clone()")
        else:
            lines.append(f"    return {output_var}")
    else:
        lines.append("    return None")

    import builtins as _builtins

    source = "\n".join(lines)
    if os.environ.get("TORCHLITE_DUMP_CODEGEN"):
        import logging as _log
        _log.getLogger(__name__).warning("Generated inference code:\n%s", source)
    code_obj = _builtins.compile(source, "<torchlite_codegen>", "exec")
    exec(code_obj, namespace)  # noqa: S102
    return namespace["forward"]


def codegen(
    gm: GraphModule,
    *,
    cuda_graphs: bool = False,
    example_inputs: Optional[List[torch.Tensor]] = None,
    inference_codegen: bool = False,
) -> Callable:
    """Convert a transformed graph into a callable.

    This is the final step of the trace → run_passes → codegen pipeline.
    When cuda_graphs=True, runs the cudagraph_partition analysis pass
    and then builds a CUDA graph callable via cudagraph_backend.
    When inference_codegen=True, generates an optimized forward function
    with pre-allocated buffers and direct Triton kernel calls.
    Requires example_inputs when cuda_graphs or inference_codegen is enabled.
    """
    if inference_codegen:
        if example_inputs is None:
            raise ValueError(
                "codegen: example_inputs required when inference_codegen=True"
            )
        return codegen_inference(gm, example_inputs)

    if not cuda_graphs:
        return gm

    if example_inputs is None:
        raise ValueError("codegen: example_inputs required when cuda_graphs=True")

    from torch._torchlite.backends import cudagraph_backend

    cudagraph_partition(gm, example_inputs)
    return cudagraph_backend(gm, example_inputs)


def compile(
    model: Union[nn.Module, Callable],
    example_inputs: List[torch.Tensor],
    lr: float = 0.01,
    optimizer_type: str = "sgd",
    dynamic_dims: Optional[Dict[str, List[int]]] = None,
    world_size: int = 1,
    cuda_graphs: bool = False,
) -> Callable:
    """Compile a model: trace → run_passes → codegen.

    Convenience wrapper equivalent to:
        codegen(run_passes(trace(model, example_inputs), example_inputs, ...))
    When cuda_graphs=True, the codegen step runs cudagraph_partition
    and wraps the result in a CUDA graph callable.
    """
    gm = trace(model, example_inputs)
    # CUDA graphs require static shapes. The default dynamize behavior marks
    # batch dim dynamic when dynamic_dims is None; disable that default for
    # cuda_graphs unless the caller explicitly requested dynamic dims.
    effective_dynamic_dims = dynamic_dims
    if cuda_graphs and dynamic_dims is None:
        effective_dynamic_dims = {}
    gm = run_passes(
        gm, example_inputs,
        lr=lr, optimizer_type=optimizer_type,
        dynamic_dims=effective_dynamic_dims, world_size=world_size,
    )
    return codegen(gm, cuda_graphs=cuda_graphs, example_inputs=example_inputs)


def precompile_save(
    gm: GraphModule,
    example_inputs: List[torch.Tensor],
    path: str,
    *,
    warmup: bool = True,
) -> None:
    """Save a compiled graph as a self-contained artifact directory.

    Runs the lowering passes (decompose → fuse → triton_codegen → precompile),
    writes the generated Python source, model state_dict, and pre-compiled
    Triton kernel binaries to disk.

    The artifact directory structure:
      path/
        compiled_module.py   - Generated Python source
        state_dict.pt        - Model parameters
        triton_cache/         - Pre-compiled Triton kernel binaries
    """
    gm = decompose(gm, example_inputs).gm
    gm = fuse(gm, example_inputs).gm
    gm = triton_codegen(gm, example_inputs).gm
    gm = precompile(gm, example_inputs).gm

    code = _graph_meta(gm.graph)["precompiled_code"]

    artifact_dir = Path(path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "compiled_module.py").write_text(code)
    torch.save(gm.state_dict(), artifact_dir / "state_dict.pt")

    # Warm the Triton cache by running the module once with example_inputs.
    # We temporarily redirect TRITON_CACHE_DIR so that exactly the needed
    # kernel binaries are captured in the artifact directory. If the
    # generated code can't run standalone (e.g. references custom ops not
    # yet supported in codegen), we skip warmup — the artifact is still
    # valid, just without pre-compiled Triton binaries.
    #
    # THREAD SAFETY WARNING: Modifying os.environ is inherently not
    # thread-safe in CPython — os.environ is a global mutable mapping and
    # concurrent reads/writes from multiple threads can race. We use
    # _triton_cache_lock to serialize concurrent precompile_save calls,
    # but this does NOT protect against other threads or libraries that
    # read or modify TRITON_CACHE_DIR independently. Do not call
    # precompile_save concurrently with other Triton cache operations.
    triton_cache_dir = str(artifact_dir / "triton_cache")

    if warmup:
        with _triton_cache_lock:
            orig_cache_dir = os.environ.get("TRITON_CACHE_DIR")
            try:
                os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
                mod_name = f"_torchlite_warmup_{uuid.uuid4().hex}"
                spec = importlib.util.spec_from_file_location(
                    mod_name, artifact_dir / "compiled_module.py"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                state_dict = torch.load(
                    artifact_dir / "state_dict.pt", weights_only=True
                )
                compiled = mod.CompiledModule(state_dict)
                compiled.forward(*example_inputs)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"precompile_save: Triton cache warmup failed ({e}). "
                    "The artifact was saved but without pre-compiled kernel binaries.",
                    stacklevel=2,
                )
                Path(triton_cache_dir).mkdir(parents=True, exist_ok=True)
            finally:
                if orig_cache_dir is None:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                else:
                    os.environ["TRITON_CACHE_DIR"] = orig_cache_dir
    else:
        Path(triton_cache_dir).mkdir(parents=True, exist_ok=True)


def precompile_load(path: str, *, trust_remote_code: bool = False) -> object:
    """Load a precompiled artifact and return a callable module.

    Dynamically imports the saved compiled_module.py, loads the state_dict,
    and merges pre-compiled Triton kernels into the active Triton cache
    so they are found without recompilation.

    WARNING: This function executes arbitrary Python code from the artifact
    directory via importlib. Only load artifacts from trusted sources.
    A malicious compiled_module.py can execute arbitrary code on load.

    Args:
        path: Path to the artifact directory created by precompile_save.
        trust_remote_code: Must be True to acknowledge that loading will
            execute arbitrary code from the artifact directory.
    """
    if not trust_remote_code:
        raise ValueError(
            "precompile_load executes arbitrary Python code from the artifact "
            "directory. Pass trust_remote_code=True to acknowledge this risk."
        )
    artifact_dir = Path(path)

    mod_name = f"_torchlite_loaded_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        mod_name, artifact_dir / "compiled_module.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    state_dict = torch.load(
        artifact_dir / "state_dict.pt", weights_only=True
    )

    # Merge pre-compiled Triton kernels into a torchlite-specific
    # subdirectory of the active Triton cache. Using a subdirectory
    # avoids overwriting the user's existing cache entries.
    triton_cache_src = artifact_dir / "triton_cache"
    if triton_cache_src.exists():
        active_cache = Path(os.environ.get(
            "TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")
        ))
        target = active_cache / "torchlite_precompiled"
        shutil.copytree(str(triton_cache_src), str(target), dirs_exist_ok=True)

    return mod.CompiledModule(state_dict)
