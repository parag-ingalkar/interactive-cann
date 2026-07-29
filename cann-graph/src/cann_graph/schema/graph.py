"""Graph specification model for CANN schema."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from cann_graph.schema.nodes import NodeSpec
from cann_graph.schema.edges import EdgeSpec


class GraphSpec(BaseModel):
    """Specification for a complete CANN energy graph."""

    schema_version: str = Field(default="1.0", description="Schema version string")
    graph_id: str = Field(..., description="Unique identifier for this graph")
    nodes: list[NodeSpec] = Field(..., description="List of node specifications")
    edges: list[EdgeSpec] = Field(..., description="List of edge specifications")
    outputs: list[str] = Field(
        ..., description="List of output node IDs (typically one for energy)"
    )
    parameters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of parameter specifications referenced by this graph"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_ids(cls, nodes: list[NodeSpec]) -> list[NodeSpec]:
        node_ids = [n.id for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            raise ValueError(f"Duplicate node IDs found: {set(duplicates)}")
        return nodes

    @field_validator("edges")
    @classmethod
    def validate_unique_edge_ids(cls, edges: list[EdgeSpec]) -> list[EdgeSpec]:
        edge_ids = [e.id for e in edges]
        if len(edge_ids) != len(set(edge_ids)):
            duplicates = [eid for eid in edge_ids if edge_ids.count(eid) > 1]
            raise ValueError(f"Duplicate edge IDs found: {set(duplicates)}")
        return edges

    @field_validator("outputs")
    @classmethod
    def validate_outputs_exist(cls, outputs: list[str], info) -> list[str]:
        # Note: We can't access nodes here directly, validation will happen in graph_validator
        return outputs

    @property
    def node_ids(self) -> set[str]:
        """Return set of all node IDs."""
        return {node.id for node in self.nodes}

    @property
    def enabled_nodes(self) -> list[NodeSpec]:
        """Return list of enabled nodes."""
        return [node for node in self.nodes if node.enabled]

    @property
    def enabled_edges(self) -> list[EdgeSpec]:
        """Return list of enabled edges."""
        return [edge for edge in self.edges if edge.enabled]

    model_config = {
        "json_schema_extra": {
            "description": "Complete graph specification with nodes, edges, and outputs"
        }
    }
