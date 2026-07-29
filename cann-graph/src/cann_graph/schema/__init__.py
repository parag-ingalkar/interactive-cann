"""Schema module for CANN graph specification."""

from cann_graph.schema.enums import (
    NodeCategory,
    ParameterConstraintType,
    RegularizerType,
    ProtocolType,
    InvariantType,
    ActivationType,
    TensorType,
)
from cann_graph.schema.parameters import ParameterSpec, ConstraintSpec, RegularizerSpec
from cann_graph.schema.nodes import NodeSpec
from cann_graph.schema.edges import EdgeSpec
from cann_graph.schema.graph import GraphSpec
from cann_graph.schema.protocol import ProtocolSpec, DEFAULT_PROTOCOLS
from cann_graph.schema.experiment import MaterialSpec, TrainingConfig, MultiStartConfig, ExperimentSpec
from cann_graph.schema.examples import (
    EXAMPLE_GRAPHS,
    create_constant_energy_graph,
    create_linear_i1_graph,
    create_quadratic_i1_graph,
    create_mooney_rivlin_graph,
    create_four_basis_expansion_graph,
)

__all__ = [
    # Enums
    "NodeCategory",
    "ParameterConstraintType",
    "RegularizerType",
    "ProtocolType",
    "InvariantType",
    "ActivationType",
    "TensorType",
    # Models
    "ParameterSpec",
    "ConstraintSpec",
    "RegularizerSpec",
    "NodeSpec",
    "EdgeSpec",
    "GraphSpec",
    "ProtocolSpec",
    "DEFAULT_PROTOCOLS",
    "MaterialSpec",
    "TrainingConfig",
    "MultiStartConfig",
    "ExperimentSpec",
    # Examples
    "EXAMPLE_GRAPHS",
    "create_constant_energy_graph",
    "create_linear_i1_graph",
    "create_quadratic_i1_graph",
    "create_mooney_rivlin_graph",
    "create_four_basis_expansion_graph",
]
