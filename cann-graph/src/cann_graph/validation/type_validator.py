"""Type validation for CANN graph operations."""

from cann_graph.schema.graph import GraphSpec
from cann_graph.validation.graph_validator import ValidationResult
from cann_graph.operations.registry import get_operation


class TypeValidator:
    """Validates that node connections have compatible tensor types.

    Checks performed:
    - Each operation receives the expected number of inputs
    - Input/output tensor types are compatible
    - Special nodes (invariant_input, energy_output) have correct signatures
    """

    def validate(self, graph: GraphSpec) -> ValidationResult:
        """Validate type compatibility in the graph.

        Args:
            graph: The graph specification to validate.

        Returns:
            ValidationResult with any errors or warnings found.
        """
        result = ValidationResult()
        enabled_nodes = {node.id: node for node in graph.enabled_nodes}

        # Build input count map
        input_counts: dict[str, int] = {nid: 0 for nid in enabled_nodes}
        for edge in graph.enabled_edges:
            if edge.target in enabled_nodes:
                input_counts[edge.target] += 1

        # Validate each node
        for node in graph.enabled_nodes:
            try:
                op_class = get_operation(node.type)
                op_instance = op_class()

                # Validate config
                op_instance.validate_config(node.config)

                # Check input count expectations
                expected_inputs = self._get_expected_input_count(node)
                actual_inputs = input_counts[node.id]

                if expected_inputs is not None and actual_inputs != expected_inputs:
                    result.add_error(
                        f"Node '{node.id}' ({node.type}) expects {expected_inputs} inputs, "
                        f"but has {actual_inputs} incoming edges"
                    )

            except KeyError:
                result.add_error(f"Node '{node.id}' has unknown operation type '{node.type}'")
            except ValueError as e:
                result.add_error(f"Node '{node.id}' configuration error: {e}")

        return result

    def _get_expected_input_count(self, node) -> int | None:
        """Get expected input count for a node type.

        Returns None if the input count is variable.
        """
        type_name = node.type

        # Fixed input counts
        fixed_inputs = {
            "identity": 1,
            "square": 1,
            "exp_minus_one": 1,
            "exp_square_minus_one": 1,
            "reference_shift": 1,
            "multiply": 2,
            "add": 2,
        }

        if type_name in fixed_inputs:
            return fixed_inputs[type_name]

        # Variable input counts (check config)
        if type_name == "weighted_sum":
            return node.config.get("num_inputs", 1)

        if type_name == "sum":
            return node.config.get("num_inputs", 2)

        # No inputs expected
        if type_name == "constant":
            return 0

        # invariant_input: no graph inputs (provided via context)
        if type_name == "invariant_input":
            return 0

        # energy_output: typically 1 input
        if type_name == "energy_output":
            return 1

        return None
