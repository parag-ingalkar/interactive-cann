"""Graph compiler for CANN energy graphs.

This module compiles a validated GraphSpec into an executable TensorFlow model
that computes scalar strain energy ψ from invariant inputs.
"""

from collections import defaultdict
from typing import Any

import tensorflow as tf

from cann_graph.schema.graph import GraphSpec
from cann_graph.schema.nodes import NodeSpec
from cann_graph.schema.edges import EdgeSpec
from cann_graph.operations.registry import get_operation
from cann_graph.compiler.parameter_store import ParameterStore


class CompiledEnergyGraph(tf.Module):
    """A compiled energy graph ready for execution.

    This callable module takes invariant context (dictionary of invariant values)
    and returns scalar strain energy ψ with shape [batch, 1].
    """

    def __init__(
        self,
        graph_spec: GraphSpec,
        parameter_store: ParameterStore,
        node_operations: dict[str, callable],
        execution_order: list[str],
    ):
        super().__init__()
        self.graph_spec = graph_spec
        self.parameter_store = parameter_store
        self.node_operations = node_operations
        self.execution_order = execution_order
        self.output_node_ids = graph_spec.outputs

    def __call__(self, invariant_context: dict[str, tf.Tensor]) -> tf.Tensor:
        """Compute strain energy ψ from invariant inputs.

        Args:
            invariant_context: Dictionary mapping invariant names to tensor values.
                Expected shape: [batch, 1] for each invariant.

        Returns:
            Scalar strain energy ψ with shape [batch, 1].
        """
        # Store computed values by node ID
        node_values: dict[str, tf.Tensor] = {}

        # Initialize invariant inputs
        for node_id, value in invariant_context.items():
            node_values[node_id] = value

        # Execute nodes in topological order
        for node_id in self.execution_order:
            if node_id in node_values:
                # Already computed (invariant input)
                continue

            op = self.node_operations.get(node_id)
            if op is None:
                raise ValueError(f"No operation compiled for node '{node_id}'")

            # Get input values from predecessor nodes
            input_values = self._get_node_inputs(node_id, node_values)

            # Execute operation
            if len(input_values) == 0:
                # No-input operation (e.g., constant)
                result = op()
            elif len(input_values) == 1:
                result = op(input_values[0])
            else:
                result = op(*input_values)

            node_values[node_id] = result

        # Return output node value(s)
        if len(self.output_node_ids) == 1:
            return node_values[self.output_node_ids[0]]
        else:
            return {nid: node_values[nid] for nid in self.output_node_ids}

    def _get_node_inputs(
        self, node_id: str, node_values: dict[str, tf.Tensor]
    ) -> list[tf.Tensor]:
        """Get input values for a node from its predecessors."""
        inputs = []
        for edge in self.graph_spec.edges:
            if edge.target == node_id and edge.enabled:
                source_id = edge.source
                if source_id not in node_values:
                    raise ValueError(
                        f"Node '{node_id}' requires input from '{source_id}' "
                        f"which has not been computed yet"
                    )
                inputs.append(node_values[source_id])
        return inputs

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        """Get all trainable variables in this graph."""
        return self.parameter_store.trainable_variables


def topological_sort(graph: GraphSpec) -> list[str]:
    """Perform topological sort on the graph nodes.

    Args:
        graph: The graph specification.

    Returns:
        List of node IDs in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    # Build adjacency list (source -> targets)
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)

    enabled_nodes = {node.id for node in graph.enabled_nodes}
    enabled_edges = graph.enabled_edges

    # Initialize in-degrees
    for node_id in enabled_nodes:
        in_degree[node_id] = 0

    # Build graph
    for edge in enabled_edges:
        if edge.source in enabled_nodes and edge.target in enabled_nodes:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

    # Kahn's algorithm
    queue = [nid for nid in enabled_nodes if in_degree[nid] == 0]
    result = []

    while queue:
        node_id = queue.pop(0)
        result.append(node_id)

        for target in adj[node_id]:
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)

    if len(result) != len(enabled_nodes):
        # Find cycle
        remaining = enabled_nodes - set(result)
        raise ValueError(
            f"Graph contains a cycle. Remaining nodes: {remaining}"
        )

    return result


class GraphCompiler:
    """Compiles GraphSpec into executable TensorFlow models."""

    def compile(
        self,
        graph_spec: GraphSpec,
        parameter_store: ParameterStore | None = None,
    ) -> CompiledEnergyGraph:
        """Compile a graph specification into an executable model.

        Args:
            graph_spec: The graph specification to compile.
            parameter_store: Optional existing parameter store. If None, creates new one.

        Returns:
            A CompiledEnergyGraph ready for execution.
        """
        if parameter_store is None:
            parameter_store = ParameterStore()

        # Topological sort
        execution_order = topological_sort(graph_spec)

        # Build operations for each node
        node_operations: dict[str, callable] = {}

        for node in graph_spec.enabled_nodes:
            op_class = get_operation(node.type)
            op_class_instance = op_class()

            # Validate config
            op_class_instance.validate_config(node.config)

            # Build the TensorFlow operation
            compiled_op = op_class_instance.build(node, parameter_store)
            node_operations[node.id] = compiled_op

        return CompiledEnergyGraph(
            graph_spec=graph_spec,
            parameter_store=parameter_store,
            node_operations=node_operations,
            execution_order=execution_order,
        )
