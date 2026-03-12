"""Functionalization pass: convert in-place ops to functional equivalents."""
import operator
from typing import List

import torch
from torch.fx import GraphModule

from torch._torchlite.passes.common import (
    _create_name,
    _set_phase,
    PassResult,
)


def _find_functional_variant(name):
    """Find the functional (non-in-place) variant of an in-place op name.

    Handles simple cases (add_ → torch.add) and compound names
    (addcmul_ → torch.addcmul, scatter_add_ → torch.Tensor.scatter_add).
    Returns None if no functional variant is found.
    """
    base = name[:-1]
    functional = getattr(torch, base, None)
    if functional is not None and callable(functional):
        return functional
    functional = getattr(torch.Tensor, base, None)
    if functional is not None and callable(functional):
        return functional
    return None


def functionalize(gm: GraphModule, example_inputs: List[torch.Tensor]) -> PassResult:
    graph = gm.graph
    mutations = []
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        if not name.endswith("_") or name.startswith("__"):
            continue
        functional = _find_functional_variant(name)
        if functional is None:
            continue

        # The first arg is the tensor mutated in-place. After switching to
        # the functional variant, all subsequent uses of that tensor must
        # reference the output of the functional op instead — otherwise
        # they would silently read the stale, pre-mutation value.
        mutated = node.args[0]
        if isinstance(mutated, torch.fx.Node):
            mutations.append((mutated, node))
            node_idx = node_order[node]
            for user in list(mutated.users):
                # Preserve topological order and semantics: only redirect
                # users that occur after the mutation. Earlier users must
                # continue to read the pre-mutation value.
                user_idx = node_order.get(user, -1)
                if user is not node and user_idx > node_idx:
                    user.replace_input_with(mutated, node)

            # When the mutated tensor is a slice (getitem) of a parent,
            # the original in-place op also mutated the parent through
            # the view aliasing. We must write the functional result back
            # into the parent so downstream ops that read the full parent
            # tensor (e.g. view(parent, ...)) see the updated values.
            if (
                mutated.op == "call_function"
                and mutated.target is operator.getitem
                and len(mutated.args) >= 2
            ):
                parent = mutated.args[0]
                key = mutated.args[1]
                with graph.inserting_after(node):
                    setitem_node = graph.call_function(
                        operator.setitem, (parent, key, node)
                    )
                    setitem_node.name = _create_name(
                        graph, "setitem_back")

        node.target = functional
        node.name = _create_name(graph, name[:-1])

    # Insert copy-back nodes: the graph body is now purely functional
    # (fusible), but we still need to reflect each mutation back into the
    # original tensor so external observers see the correct state.
    # Walk mutations to find each base tensor's final value. When a
    # tensor is mutated multiple times (a.add_(b); a.add_(c)), the
    # chain is: base=a → first_result → second_result. We only need
    # one copy_(base, last_result) per base.
    if mutations:
        output_node = None
        for n in graph.nodes:
            if n.op == "output":
                output_node = n
                break
        graph.inserting_before(output_node)
        node_to_base = {}
        final_for_base = {}
        for mutated, result in mutations:
            base = node_to_base.get(mutated, mutated)
            node_to_base[result] = base
            final_for_base[base] = result
        for base, final_value in final_for_base.items():
            copy_node = graph.call_function(
                torch.Tensor.copy_, (base, final_value)
            )
            copy_node.name = _create_name(graph, "copy_back")
            _set_phase(copy_node, "copy-back")

    graph.lint()
    gm.recompile()
    return PassResult(gm=gm)
