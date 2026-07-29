"""Arithmetic and energy primitive operations for CANN graph."""

from typing import Any, Callable

import tensorflow as tf
import sympy as sp

from cann_graph.operations.base import Operation, TensorType, CompiledOperation
from cann_graph.operations.registry import register_operation
from cann_graph.schema.nodes import NodeSpec
from cann_graph.compiler.parameter_store import ParameterStore


@register_operation
class IdentityOp(Operation):
    """Identity operation: f(x) = x."""

    type_name = "identity"
    category = "energy"
    description = "Identity function"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 1:
            raise ValueError("Identity expects exactly one input")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor) -> tf.Tensor:
            return tf.identity(x)
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return inputs[0]


@register_operation
class SquareOp(Operation):
    """Square operation: f(x) = x²."""

    type_name = "square"
    category = "energy"
    description = "Square function"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 1:
            raise ValueError("Square expects exactly one input")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor) -> tf.Tensor:
            return tf.square(x)
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return inputs[0] ** 2


@register_operation
class ExpMinusOneOp(Operation):
    """Exponential minus one: f(x) = exp(x) - 1."""

    type_name = "exp_minus_one"
    category = "energy"
    description = "Exponential minus one"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 1:
            raise ValueError("ExpMinusOne expects exactly one input")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor) -> tf.Tensor:
            return tf.exp(x) - tf.ones_like(x)
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return sp.exp(inputs[0]) - 1


@register_operation
class ExpSquareMinusOneOp(Operation):
    """Exponential of square minus one: f(x) = exp(x²) - 1."""

    type_name = "exp_square_minus_one"
    category = "energy"
    description = "Exponential of square minus one"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 1:
            raise ValueError("ExpSquareMinusOne expects exactly one input")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor) -> tf.Tensor:
            return tf.exp(tf.square(x)) - tf.ones_like(x)
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return sp.exp(inputs[0] ** 2) - 1


@register_operation
class WeightedSumOp(Operation):
    """Weighted sum: z = Σ w_i * x_i.

    Supports either a single weight (for single input) or multiple weights.
    """

    type_name = "weighted_sum"
    category = "energy"
    description = "Weighted sum of inputs"

    def validate_config(self, config: dict[str, Any]) -> None:
        num_inputs = config.get("num_inputs", 1)
        if not isinstance(num_inputs, int) or num_inputs < 1:
            raise ValueError("num_inputs must be a positive integer")

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) < 1:
            raise ValueError("WeightedSum expects at least one input")
        # Output has same type as inputs (scalar)
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        num_inputs = node.config.get("num_inputs", 1)
        param_refs = node.parameter_refs

        if len(param_refs) == 1 and num_inputs == 1:
            # Single weight case
            weight = parameters.get(param_refs[0])
            def op(x: tf.Tensor) -> tf.Tensor:
                return weight * x
            return op
        elif len(param_refs) == num_inputs:
            # Multiple weights case
            weights = [parameters.get(ref) for ref in param_refs]
            def op(*inputs: tf.Tensor) -> tf.Tensor:
                result = tf.zeros_like(inputs[0])
                for i, inp in enumerate(inputs):
                    result = result + weights[i] * inp
                return result
            return op
        else:
            raise ValueError(
                f"WeightedSum with {num_inputs} inputs expects {num_inputs} weight parameters, "
                f"got {len(param_refs)}"
            )

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        param_refs = node.parameter_refs
        if len(param_refs) == 1 and len(inputs) == 1:
            w = sp.Symbol(param_refs[0])
            return w * inputs[0]
        elif len(param_refs) == len(inputs):
            result = 0
            for i, inp in enumerate(inputs):
                w = sp.Symbol(param_refs[i])
                result = result + w * inp
            return result
        else:
            raise ValueError("Mismatch between weights and inputs in symbolic")


@register_operation
class SumOp(Operation):
    """Sum of inputs: z = Σ x_i."""

    type_name = "sum"
    category = "energy"
    description = "Sum of inputs"

    def validate_config(self, config: dict[str, Any]) -> None:
        num_inputs = config.get("num_inputs", 2)
        if not isinstance(num_inputs, int) or num_inputs < 2:
            raise ValueError("num_inputs must be an integer >= 2")

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) < 2:
            raise ValueError("Sum expects at least two inputs")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(*inputs: tf.Tensor) -> tf.Tensor:
            result = inputs[0]
            for inp in inputs[1:]:
                result = result + inp
            return result
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return sum(inputs)


@register_operation
class ReferenceShiftOp(Operation):
    """Reference shift: f(x) = x - reference.

    Used to enforce stress-free reference configuration.
    """

    type_name = "reference_shift"
    category = "invariant"
    description = "Subtract reference value"

    def validate_config(self, config: dict[str, Any]) -> None:
        if "reference" not in config:
            raise ValueError("ReferenceShift requires 'reference' in config")
        ref = config["reference"]
        if not isinstance(ref, (int, float)):
            raise ValueError("Reference must be a scalar number")

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 1:
            raise ValueError("ReferenceShift expects exactly one input")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        reference = tf.constant(node.config["reference"], dtype=tf.float64)
        def op(x: tf.Tensor) -> tf.Tensor:
            return x - reference
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        reference = node.config["reference"]
        return inputs[0] - reference


@register_operation
class ConstantOp(Operation):
    """Constant value output."""

    type_name = "constant"
    category = "input"
    description = "Constant scalar value"

    def validate_config(self, config: dict[str, Any]) -> None:
        if "value" not in config:
            raise ValueError("Constant requires 'value' in config")

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        return TensorType(dtype=tf.float64, shape=())

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        value = tf.constant(node.config["value"], dtype=tf.float64)
        def op() -> tf.Tensor:
            return value
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return sp.Float(node.config["value"])


@register_operation
class MultiplyOp(Operation):
    """Element-wise multiplication: z = x * y."""

    type_name = "multiply"
    category = "energy"
    description = "Multiply two inputs"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 2:
            raise ValueError("Multiply expects exactly two inputs")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return x * y
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return inputs[0] * inputs[1]


@register_operation
class AddOp(Operation):
    """Addition: z = x + y."""

    type_name = "add"
    category = "energy"
    description = "Add two inputs"

    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        if len(input_types) != 2:
            raise ValueError("Add expects exactly two inputs")
        return input_types[0]

    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        def op(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return x + y
        return op

    def symbolic(self, inputs: list[sp.Expr], node: NodeSpec) -> sp.Expr:
        return inputs[0] + inputs[1]
