"""Base classes for CANN graph operations."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import tensorflow as tf
import sympy as sp

from cann_graph.schema.nodes import NodeSpec
from cann_graph.compiler.parameter_store import ParameterStore


class TensorType:
    """Represents the type signature of a tensor."""

    def __init__(self, dtype: tf.DType = tf.float64, shape: tuple[int | None, ...] | None = None):
        self.dtype = dtype
        self.shape = shape

    def __repr__(self) -> str:
        shape_str = "?" if self.shape is None else str(self.shape)
        return f"Tensor({self.dtype}, {shape_str})"


class CompiledOperation(Protocol):
    """Protocol for a compiled operation ready for execution."""

    def __call__(self, *inputs: tf.Tensor) -> tf.Tensor:
        """Execute the operation on input tensors."""
        ...


class SymbolicExpr(Protocol):
    """Protocol for symbolic expressions (SymPy compatibility)."""

    def __repr__(self) -> str:
        ...


class Operation(ABC):
    """Abstract base class for all CANN graph operations.

    Each operation must provide:
    - A unique type_name
    - Input/output type inference
    - TensorFlow implementation (build method)
    - Symbolic expression generation
    """

    type_name: str = "base"
    category: str = "base"
    description: str = "Base operation"

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate the node configuration for this operation.

        Args:
            config: The node's config dictionary.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def infer_output_type(self, input_types: list[TensorType]) -> TensorType:
        """Infer the output tensor type from input types.

        Args:
            input_types: List of input tensor types.

        Returns:
            The inferred output tensor type.
        """
        pass

    @abstractmethod
    def build(
        self, node: NodeSpec, parameters: ParameterStore
    ) -> CompiledOperation:
        """Build the TensorFlow operation.

        Args:
            node: The node specification.
            parameters: The parameter store for accessing shared variables.

        Returns:
            A callable TensorFlow operation.
        """
        pass

    @abstractmethod
    def symbolic(
        self, inputs: list[sp.Expr], node: NodeSpec
    ) -> sp.Expr:
        """Generate symbolic expression.

        Args:
            inputs: List of symbolic input expressions.
            node: The node specification.

        Returns:
            A SymPy expression representing the operation.
        """
        pass

    def get_parameter_names(self, node: NodeSpec) -> list[str]:
        """Get the list of parameter names used by this operation.

        Args:
            node: The node specification.

        Returns:
            List of parameter names referenced in the node config.
        """
        return node.config.get("parameter_refs", [])
