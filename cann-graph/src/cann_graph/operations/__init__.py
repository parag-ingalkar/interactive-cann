"""Operations module for CANN graph."""

from cann_graph.operations.base import Operation, TensorType, CompiledOperation
from cann_graph.operations.registry import (
    OperationRegistry,
    register_operation,
    get_operation,
)
from cann_graph.operations.arithmetic import (
    IdentityOp,
    SquareOp,
    ExpMinusOneOp,
    ExpSquareMinusOneOp,
    WeightedSumOp,
    SumOp,
    ReferenceShiftOp,
    ConstantOp,
    MultiplyOp,
    AddOp,
)

__all__ = [
    # Base classes
    "Operation",
    "TensorType",
    "CompiledOperation",
    # Registry
    "OperationRegistry",
    "register_operation",
    "get_operation",
    # Arithmetic operations
    "IdentityOp",
    "SquareOp",
    "ExpMinusOneOp",
    "ExpSquareMinusOneOp",
    "WeightedSumOp",
    "SumOp",
    "ReferenceShiftOp",
    "ConstantOp",
    "MultiplyOp",
    "AddOp",
]
