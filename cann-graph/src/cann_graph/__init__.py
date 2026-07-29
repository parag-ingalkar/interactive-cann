"""CANN Graph: A validated neuron-level graph schema compiler for shared multi-protocol CANN models."""

__version__ = "0.1.0"

from cann_graph.schema.enums import (
    NodeCategory,
    ParameterConstraintType,
    RegularizerType,
    ProtocolType,
    InvariantType,
)
from cann_graph.schema.parameters import ParameterSpec, ConstraintSpec, RegularizerSpec
from cann_graph.schema.nodes import NodeSpec
from cann_graph.schema.edges import EdgeSpec
from cann_graph.schema.graph import GraphSpec
from cann_graph.schema.protocol import ProtocolSpec
from cann_graph.schema.experiment import ExperimentSpec, MaterialSpec

from cann_graph.compiler.parameter_store import ParameterStore
from cann_graph.compiler.graph_compiler import GraphCompiler
from cann_graph.compiler.material_model import MaterialModel
from cann_graph.compiler.protocol_branch import CompiledProtocolBranch
from cann_graph.compiler.experiment_model import CANNExperimentModel

from cann_graph.operations.registry import OperationRegistry, get_operation, register_operation
from cann_graph.operations.base import CompiledOperation

__all__ = [
    # Version
    "__version__",
    # Schema enums
    "NodeCategory",
    "ParameterConstraintType",
    "RegularizerType",
    "ProtocolType",
    "InvariantType",
    # Schema models
    "ParameterSpec",
    "ConstraintSpec",
    "RegularizerSpec",
    "NodeSpec",
    "EdgeSpec",
    "GraphSpec",
    "ProtocolSpec",
    "ExperimentSpec",
    "MaterialSpec",
    # Compiler
    "ParameterStore",
    "GraphCompiler",
    "MaterialModel",
    "CompiledProtocolBranch",
    "CANNExperimentModel",
    # Operations
    "OperationRegistry",
    "get_operation",
    "register_operation",
    "CompiledOperation",
]
