"""Enumeration types for CANN graph schema."""

from enum import Enum


class NodeCategory(str, Enum):
    """Categories of nodes in the CANN graph."""

    INPUT = "input"
    MECHANICS = "mechanics"
    INVARIANT = "invariant"
    DISPERSION = "dispersion"
    ENERGY = "energy"
    OUTPUT = "output"
    UTILITY = "utility"


class ParameterConstraintType(str, Enum):
    """Types of parameter constraints."""

    INTERVAL = "interval"
    NON_NEGATIVE = "non_negative"
    POSITIVE = "positive"
    UNIT_INTERVAL = "unit_interval"


class RegularizerType(str, Enum):
    """Types of regularizers for parameters."""

    L1 = "l1"
    L2 = "l2"
    NONE = "none"


class ProtocolType(str, Enum):
    """Types of experimental protocols."""

    UTC_FIBER = "utc_fiber"
    UTC_SHEET = "utc_sheet"
    UTC_NORMAL = "utc_normal"
    SS_FIBER = "ss_fiber"
    SS_SHEET = "ss_sheet"
    SS_NORMAL = "ss_normal"
    FS_FS = "fs_fs"
    FS_FN = "fs_fn"
    FS_SN = "fs_sn"


class InvariantType(str, Enum):
    """Types of strain invariants."""

    I1 = "I1"
    I2 = "I2"
    I4F = "I4f"
    I4S = "I4s"
    I4N = "I4n"
    I8FS = "I8fs"
    I8FN = "I8fn"
    I8SN = "I8sn"


class ActivationType(str, Enum):
    """Types of activation functions for energy neurons."""

    IDENTITY = "identity"
    SQUARE = "square"
    EXP_MINUS_ONE = "exp_minus_one"
    EXP_SQUARE_MINUS_ONE = "exp_square_minus_one"


class TensorType(str, Enum):
    """Tensor type signatures for operation validation."""

    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX_3X3 = "matrix_3x3"
    TENSOR_RANK2 = "tensor_rank2"
