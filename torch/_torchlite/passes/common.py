"""Common utilities and data structures for torchlite passes."""
import logging
import operator
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule
from torch.overrides import resolve_name

from torch._torchlite.ops import (
    _save_for_backward,
    _save_rng_state,
    _load_rng_state,
    _UNARY_POINTWISE_OPS,
    adamw_step,
    param_update,
)

log = logging.getLogger(__name__)


@dataclass
class PassResult:
    gm: GraphModule
    changed: bool = True


# WeakKeyDictionary: metadata is keyed by Graph objects. If a Graph is
# garbage-collected (e.g. the GraphModule goes out of scope), its metadata
# is silently dropped. This can cause the optimizer pass to silently skip
# parameter updates if the graph that autograd_per_op wrote to has been
# collected and re-created. If you see optimizer() returning a no-op when
# it shouldn't, check that the GraphModule hasn't been re-traced.
_graph_meta_store: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _create_name(graph, name):
    return graph._graph_namespace.create_name(name, None)


def _graph_meta(graph):
    meta = _graph_meta_store.get(graph)
    if meta is None:
        meta = {}
        _graph_meta_store[graph] = meta
    return meta


@dataclass
class FusionGroup:
    group_id: int
    node_names: List[str]
    op_names: List[str]
    shape: Optional[List[int]]
    inputs: List[str]
    output: str



@dataclass
class FusedOp:
    op_name: str
    args: List  # each is ("input", idx), ("tmp", idx), or ("const", value)


# eq=False: FusedKernel instances are used as FX node targets and must
# be compared by identity, not by value. Two kernels with the same ops
# but different graph positions are distinct targets.
@dataclass(eq=False)
class FusedKernel:
    name: str
    ops: List[FusedOp]
    n_inputs: int
    shape: Optional[List[int]] = None
    input_shapes: Optional[List[Optional[List[int]]]] = None

    def __post_init__(self):
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.name}: fused kernel placeholder — run triton_codegen "
            f"and precompile passes, then load the generated module"
        )


@dataclass(eq=False)
class MatmulEpilogueKernel:
    name: str
    epilogue_ops: List[FusedOp]
    has_bias: bool
    M: int
    N: int
    K: int
    dtype: torch.dtype
    extra_shapes: List[tuple] = field(default_factory=list)
    # When the epilogue operates on a restored ND shape (e.g. [B, S, N]
    # from a 3D input batch), out_shape records that shape so the kernel
    # runner can allocate and reshape the output correctly.  None means
    # the output is the raw 2D mm shape [M, N].
    out_shape: Optional[List[int]] = None

    def __post_init__(self):
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.name}: matmul epilogue placeholder — run triton_lower "
            f"to compile into a Triton kernel"
        )


@dataclass(eq=False)
class AddRmsNormKernel:
    """Fused add + rms_norm kernel placeholder.

    Pattern: c = a + b; out = rms_norm(c, weight, eps).
    Produces two outputs (add_result, norm_result) accessed via getitem.
    The Triton kernel computes both in a single pass over the data,
    avoiding a separate kernel launch for the elementwise add.

    When has_add=False, this is a standalone rms_norm (no preceding add).
    The kernel takes (input, weight) instead of (a, b, weight) and
    produces a single output (the norm result) rather than a tuple.
    """
    name: str
    shape: List[int]
    norm_dim: int
    eps: float
    dtype: torch.dtype
    has_add: bool = True

    def __post_init__(self):
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.name}: add+rms_norm placeholder — run triton_lower "
            f"to compile into a Triton kernel"
        )


@dataclass(eq=False)
class AddLayerNormKernel:
    """Fused add + layer_norm kernel placeholder.

    Pattern: c = a + b; out = layer_norm(c, shape, weight, bias, eps).
    Produces two outputs (add_result, norm_result) accessed via getitem.
    LayerNorm computes mean and variance (unlike RMSNorm which only uses
    variance), so the Triton kernel has a slightly different reduction.

    When has_add=False, this is a standalone layer_norm (no preceding add).
    The kernel takes (input, weight, bias) and produces a single output.
    """
    name: str
    shape: List[int]
    norm_dim: int
    eps: float
    dtype: torch.dtype
    has_add: bool = True
    has_norm_bias: bool = False

    def __post_init__(self):
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args):
        raise NotImplementedError(
            f"{self.name}: add+layer_norm placeholder — run triton_lower "
            f"to compile into a Triton kernel"
        )


_VARARGS_TENSOR_METHODS = frozenset({
    "reshape", "view", "expand", "repeat", "permute",
    "flip", "squeeze",
})

_DUNDER_TO_OP = {
    "__add__": "add",
    "__sub__": "sub",
    "__mul__": "mul",
    "__matmul__": "matmul",
    "__truediv__": "div",
    "__floordiv__": "floor_divide",
    "__mod__": "remainder",
    "__pow__": "pow",
    "__neg__": "neg",
    "__abs__": "abs",
    "__eq__": "eq",
    "__ne__": "ne",
    "__lt__": "lt",
    "__le__": "le",
    "__gt__": "gt",
    "__ge__": "ge",
    "__invert__": "bitwise_not",
    "__and__": "bitwise_and",
    "__or__": "bitwise_or",
    "__xor__": "bitwise_xor",
}

_REVERSE_DUNDERS = {
    "__radd__", "__rsub__", "__rmul__", "__rmatmul__",
    "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__",
    "__rand__", "__ror__", "__rxor__",
}

_DUNDER_INPLACE = {
    "__iadd__": "add_",
    "__isub__": "sub_",
    "__imul__": "mul_",
    "__itruediv__": "div_",
    "__iand__": "bitwise_and_",
    "__ior__": "bitwise_or_",
    "__ixor__": "bitwise_xor_",
}


def _set_phase(node, phase):
    node.meta["phase"] = phase


def _deep_getattr(obj, target):
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _deep_setattr(obj, target, value):
    parts = target.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            setattr(obj, part, torch.nn.Module())
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


_PROVENANCE_KEYS = frozenset({"phase", "bwd_of", "dtensor_spec", "rng_replay_for"})


def _is_torch_op(target):
    if target is operator.getitem:
        return False
    if isinstance(target, (FusedKernel, MatmulEpilogueKernel, AddRmsNormKernel, AddLayerNormKernel)):
        return False
    if isinstance(target, torch._ops.OpOverload):
        return True
    module = getattr(target, "__module__", "") or ""
    if "torchlite" in module:
        return False
    return module.startswith("torch")


def _iter_node_args(node):
    for a in node.args:
        if isinstance(a, (list, tuple)):
            yield from a
        else:
            yield a
    for v in node.kwargs.values():
        if isinstance(v, (list, tuple)):
            yield from v
        else:
            yield v


def _aten_op_name(target):
    packet = getattr(target, "overloadpacket", None)
    if packet is not None:
        return getattr(packet, "__name__", str(target))
    return getattr(target, "__name__", str(target))


def _node_shape(node):
    return node.meta.get("shape")
