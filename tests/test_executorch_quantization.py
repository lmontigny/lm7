"""Graph repairs LM7 applies before PT2E INT8 quantization.

These exercise `lm7.backends.executorch` helpers that only need `torch.fx`, so
they run wherever the normal test suite runs -- ExecuTorch itself is only needed
for the end-to-end export in `test_executorch_integration.py`.
"""

from __future__ import annotations

import torch

from lm7.backends.executorch import _is_floating, _retype_integer_scalar_lifts


def _graph_with_lifted_scalar(
    *,
    operand: torch.Tensor,
    constant: torch.Tensor,
    result: torch.Tensor,
) -> torch.fx.GraphModule:
    """A one-op graph shaped like what `transform_for_annotation` produces.

    PT2E rewrites a literal scalar into a `get_attr` tensor constant so it can be
    observed, which is the rewrite these helpers have to cope with.
    """
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("ids")
    placeholder.meta["val"] = operand
    lifted = graph.get_attr("_tensor_constant_0")
    lifted.meta["val"] = constant
    added = graph.call_function(torch.ops.aten.add.Tensor, (placeholder, lifted))
    added.meta["val"] = result
    graph.output((added,))

    root = torch.nn.Module()
    root._tensor_constant_0 = constant
    return torch.fx.GraphModule(root, graph)


def test_integer_scalar_lift_is_retyped_to_the_operand_dtype():
    """The bug: `input_ids + 1` becomes int64 + float32, which promotes to float.

    A float index then fails the embedding lookup further down the graph.
    """
    module = _graph_with_lifted_scalar(
        operand=torch.zeros(1, 4, dtype=torch.int64),
        constant=torch.tensor(1.0),
        result=torch.zeros(1, 4, dtype=torch.int64),
    )
    assert module._tensor_constant_0.dtype is torch.float32

    assert _retype_integer_scalar_lifts(module) == 1

    assert module._tensor_constant_0.dtype is torch.int64
    assert int(module._tensor_constant_0) == 1


def test_float_operands_keep_their_scalar():
    module = _graph_with_lifted_scalar(
        operand=torch.zeros(1, 4),
        constant=torch.tensor(1.0),
        result=torch.zeros(1, 4),
    )

    assert _retype_integer_scalar_lifts(module) == 0
    assert module._tensor_constant_0.dtype is torch.float32


def test_a_fractional_scalar_is_left_alone():
    """Truncating 0.5 to 0 would change the model rather than repair it."""
    module = _graph_with_lifted_scalar(
        operand=torch.zeros(1, 4, dtype=torch.int64),
        constant=torch.tensor(0.5),
        result=torch.zeros(1, 4),
    )

    assert _retype_integer_scalar_lifts(module) == 0
    assert module._tensor_constant_0.dtype is torch.float32


def test_is_floating_defaults_to_true_without_metadata():
    """Unknown nodes must not be stripped of quantization by accident."""
    graph = torch.fx.Graph()
    node = graph.placeholder("x")
    graph.output((node,))

    assert _is_floating(node) is True

    node.meta["val"] = torch.zeros(2, dtype=torch.int64)
    assert _is_floating(node) is False

    node.meta["val"] = torch.zeros(2)
    assert _is_floating(node) is True
