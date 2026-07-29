"""Experiment specification models for CANN."""

from typing import Any

from pydantic import BaseModel, Field

from cann_graph.schema.protocol import ProtocolSpec
from cann_graph.schema.graph import GraphSpec
from cann_graph.schema.parameters import ParameterSpec


class MaterialSpec(BaseModel):
    """Specification for material parameters and configuration."""

    parameters: list[ParameterSpec] = Field(
        ..., description="List of material parameter specifications"
    )
    energy_graph_id: str = Field(
        ..., description="ID of the energy graph used by this material"
    )
    fixed_theta: bool = Field(
        default=False, description="Whether the structural angle theta is fixed"
    )
    fixed_kappa: bool = Field(
        default=False, description="Whether the dispersion parameter kappa is fixed"
    )
    tension_only: bool = Field(
        default=True, description="Whether to use tension-only activation"
    )
    reference_shift: bool = Field(
        default=True, description="Whether to apply stress-free reference shift"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Material specification with parameters and configuration"
        }
    }


class TrainingConfig(BaseModel):
    """Configuration for training the CANN model."""

    optimizer: str = Field(default="adam", description="Optimizer type")
    learning_rate: float = Field(default=0.01, description="Initial learning rate")
    lr_schedule: str = Field(
        default="reduce_on_plateau", description="Learning rate schedule type"
    )
    epochs: int = Field(default=1000, description="Maximum number of epochs")
    batch_size: int | None = Field(
        default=None, description="Batch size (None for full-batch)"
    )
    l1_scale: float = Field(default=0.0, description="L1 regularization scale")
    early_stopping_patience: int = Field(
        default=50, description="Early stopping patience"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {
        "json_schema_extra": {
            "description": "Training configuration"
        }
    }


class MultiStartConfig(BaseModel):
    """Configuration for multi-start training."""

    parameter_initializations: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Parameter IDs and their initial values to try"
    )
    selection_metric: str = Field(
        default="mean_protocol_r2", description="Metric for selecting best run"
    )
    num_starts: int | None = Field(
        default=None, description="Number of starts (if not using explicit initializations)"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Multi-start configuration"
        }
    }


class ExperimentSpec(BaseModel):
    """Complete specification for a CANN experiment."""

    experiment_id: str = Field(..., description="Unique identifier for this experiment")
    material: MaterialSpec = Field(..., description="Material specification")
    protocols: list[ProtocolSpec] = Field(..., description="Protocol specifications")
    energy_graph: GraphSpec = Field(..., description="Energy graph specification")
    training: TrainingConfig | None = Field(
        default=None, description="Training configuration"
    )
    multistart: MultiStartConfig | None = Field(
        default=None, description="Multi-start configuration"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional experiment metadata"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Complete experiment specification"
        }
    }
