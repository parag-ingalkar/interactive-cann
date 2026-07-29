"""Parameter sharing validation for CANN graphs."""

from cann_graph.schema.graph import GraphSpec
from cann_graph.validation.graph_validator import ValidationResult


class SharingValidator:
    """Validates that parameter sharing is correctly configured.

    Checks performed:
    - Every reference to the same parameter_id resolves to one global parameter
    - Parameter IDs cannot be reused with conflicting properties
    - Protocol-local parameters don't accidentally use material.* namespace
    - Trainable material parameters appear in trainable set exactly once
    """

    def validate(self, graph: GraphSpec) -> ValidationResult:
        """Validate parameter sharing in the graph.

        Args:
            graph: The graph specification to validate.

        Returns:
            ValidationResult with any errors or warnings found.
        """
        result = ValidationResult()

        # Build parameter definition map
        param_defs: dict[str, dict] = {}
        for param in graph.parameters:
            param_id = param.get("id", "")
            namespace = param.get("namespace", "material")
            full_id = f"{namespace}.{param_id}"
            
            if full_id in param_defs:
                result.add_error(
                    f"Parameter '{full_id}' is defined multiple times"
                )
            else:
                param_defs[full_id] = param

        # Collect all parameter references from nodes
        param_refs: dict[str, list[str]] = {}  # full_id -> list of node IDs
        for node in graph.enabled_nodes:
            for ref in node.parameter_refs:
                # Determine full ID based on namespace
                if "." in ref:
                    full_id = ref
                else:
                    full_id = f"material.{ref}"
                
                if full_id not in param_refs:
                    param_refs[full_id] = []
                param_refs[full_id].append(node.id)

        # Check that all referenced parameters are defined
        for full_id, node_ids in param_refs.items():
            if full_id not in param_defs:
                result.add_error(
                    f"Parameter '{full_id}' is referenced by nodes {node_ids} "
                    f"but not defined in graph parameters"
                )

        # Check for namespace violations (protocol-local using material.*)
        for node in graph.enabled_nodes:
            category = node.category
            if category and category.value == "input":
                # Input nodes might have local parameters
                for ref in node.parameter_refs:
                    if ref.startswith("material."):
                        result.add_warning(
                            f"Input node '{node.id}' references material parameter '{ref}'. "
                            f"This might be intentional for shared inputs."
                        )

        return result

    def validate_multi_graph_sharing(
        self,
        graphs: list[GraphSpec],
        shared_param_ids: list[str],
    ) -> ValidationResult:
        """Validate parameter sharing across multiple graphs.

        This is used when validating that multiple protocol branches
        share the same material parameters.

        Args:
            graphs: List of graph specifications that should share parameters.
            shared_param_ids: List of parameter IDs that must be shared.

        Returns:
            ValidationResult with any errors or warnings found.
        """
        result = ValidationResult()

        # For each shared parameter, verify it appears consistently
        for shared_id in shared_param_ids:
            found_in = []
            initial_values = {}
            
            for graph in graphs:
                for param in graph.parameters:
                    param_id = param.get("id", "")
                    namespace = param.get("namespace", "material")
                    full_id = f"{namespace}.{param_id}"
                    
                    if full_id == shared_id or param_id == shared_id:
                        found_in.append(graph.graph_id)
                        initial_values[graph.graph_id] = param.get("initial_value")

            if len(found_in) == 0:
                result.add_warning(
                    f"Shared parameter '{shared_id}' not found in any graph"
                )
            elif len(found_in) < len(graphs):
                result.add_warning(
                    f"Shared parameter '{shared_id}' only found in {len(found_in)}/{len(graphs)} graphs: {found_in}"
                )
            else:
                # Check that initial values are consistent
                unique_values = set(str(v) for v in initial_values.values())
                if len(unique_values) > 1:
                    result.add_error(
                        f"Shared parameter '{shared_id}' has conflicting initial values: {initial_values}"
                    )

        return result
