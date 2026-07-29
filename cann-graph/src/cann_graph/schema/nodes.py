"""Node specification models for CANN graph schema."""

from typing import Any

from pydantic import BaseModel, Field

from cann_graph.schema.enums import NodeCategory


class NodeSpec(BaseModel):
    """Specification for a node in the CANN graph."""

    id: str = Field(..., description="Unique identifier for this node")
    type: str = Field(..., description="Operation type (must be registered)")
    label: str | None = Field(default=None, description="Human-readable label")
    category: NodeCategory | None = Field(
        default=None, description="Node category for organization"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific configuration"
    )
    enabled: bool = Field(default=True, description="Whether this node is active")
    parameter_refs: list[str] = Field(
        default_factory=list,
        description="List of parameter IDs referenced by this node"
    )

    model_config = {
        "json_schema_extra": {
            "description": "Node specification with type, configuration, and parameter references"
        }
    }

    def get_parameter_ref(self, param_name: str) -> str | None:
        """Get a parameter reference by name from the node's config."""
        return self.config.get(f"{param_name}_ref")
