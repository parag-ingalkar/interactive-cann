"""Edge specification models for CANN graph schema."""

from pydantic import BaseModel, Field


class EdgeSpec(BaseModel):
    """Specification for an edge (connection) between nodes in the CANN graph."""

    id: str = Field(..., description="Unique identifier for this edge")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    target_port: str = Field(
        default="input", description="Target port name (for multi-input nodes)"
    )
    enabled: bool = Field(default=True, description="Whether this edge is active")
    weight: float | None = Field(
        default=None, description="Optional edge weight (for weighted connections)"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Edge specification connecting source and target nodes"
        }
    }
