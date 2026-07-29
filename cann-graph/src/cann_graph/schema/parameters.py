"""Parameter specification models for CANN graph schema."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from cann_graph.schema.enums import ParameterConstraintType, RegularizerType


class ConstraintSpec(BaseModel):
    """Specification for a parameter constraint."""

    type: ParameterConstraintType
    min_value: float | None = None
    max_value: float | None = None

    @field_validator("min_value", "max_value")
    @classmethod
    def validate_interval_bounds(cls, v: float | None, info) -> float | None:
        if info.data.get("type") == ParameterConstraintType.INTERVAL:
            if v is None:
                raise ValueError("Interval constraint requires min_value and max_value")
        return v

    model_config = {
        "json_schema_extra": {
            "description": "Constraint specification for trainable parameters"
        }
    }


class RegularizerSpec(BaseModel):
    """Specification for a parameter regularizer."""

    type: RegularizerType
    scale: float = 1.0

    model_config = {
        "json_schema_extra": {
            "description": "Regularizer specification for training"
        }
    }


class ParameterSpec(BaseModel):
    """Specification for a trainable or fixed parameter in the CANN graph."""

    id: str = Field(..., description="Unique identifier for this parameter")
    initial_value: float | list[float] = Field(
        ..., description="Initial value(s) for the parameter"
    )
    trainable: bool = Field(
        default=True, description="Whether this parameter should be optimized"
    )
    constraint: ConstraintSpec | None = Field(
        default=None, description="Optional constraint on parameter values"
    )
    regularizer: RegularizerSpec | None = Field(
        default=None, description="Optional regularizer for training"
    )
    dtype: Literal["float64"] = Field(
        default="float64", description="Data type (must be float64 for constitutive modeling)"
    )
    namespace: str = Field(
        default="material",
        description="Namespace for parameter sharing (e.g., 'material', 'protocol_local')"
    )

    @property
    def full_id(self) -> str:
        """Return the fully qualified parameter ID including namespace."""
        return f"{self.namespace}.{self.id}"

    @field_validator("initial_value")
    @classmethod
    def validate_initial_value(cls, v: float | list[float]) -> float | list[float]:
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("initial_value list cannot be empty")
        return v

    model_config = {
        "json_schema_extra": {
            "description": "Parameter specification with sharing, constraints, and regularization"
        }
    }
