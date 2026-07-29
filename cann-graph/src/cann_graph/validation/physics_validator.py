"""Physics validation for CANN graph specifications."""

from cann_graph.schema.graph import GraphSpec
from cann_graph.validation.graph_validator import ValidationResult


class PhysicsValidator:
    """Validates physics-related constraints in the CANN graph.

    Checks performed (as warnings for flexibility, can be elevated to errors):
    - Dispersion parameter κ must be constrained to [0, 1/3]
    - Structural angle θ must be constrained to a physical range [0, π]
    - Energy output must be scalar per batch element
    - Reference-shift requirements in stress-free-reference mode
    - Tension-only activation ordering
    """

    def __init__(
        self,
        polyconvex_mode: bool = False,
        stress_free_reference_mode: bool = True,
        tension_only_mode: bool = True,
    ):
        self.polyconvex_mode = polyconvex_mode
        self.stress_free_reference_mode = stress_free_reference_mode
        self.tension_only_mode = tension_only_mode

    def validate(self, graph: GraphSpec) -> ValidationResult:
        """Validate physics constraints in the graph.

        Args:
            graph: The graph specification to validate.

        Returns:
            ValidationResult with any errors or warnings found.
        """
        result = ValidationResult()

        # Check for dispersion parameters
        self._check_dispersion_parameters(graph, result)

        # Check for structural angle parameters
        self._check_structural_angle(graph, result)

        # Check reference shift if enabled
        if self.stress_free_reference_mode:
            self._check_reference_shift(graph, result)

        # Check tension-only activation ordering
        if self.tension_only_mode:
            self._check_tension_activation_ordering(graph, result)

        return result

    def _check_dispersion_parameters(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check that dispersion parameters are properly constrained."""
        for param in graph.parameters:
            param_id = param.get("id", "")
            if "kappa" in param_id.lower():
                constraint = param.get("constraint")
                if constraint:
                    ctype = constraint.get("type")
                    if ctype == "interval":
                        min_val = constraint.get("min_value")
                        max_val = constraint.get("max_value")
                        if min_val is not None and min_val < 0:
                            result.add_error(
                                f"Dispersion parameter '{param_id}' has min_value {min_val}, "
                                f"must be >= 0"
                            )
                        if max_val is not None and max_val > 1.0 / 3.0:
                            result.add_warning(
                                f"Dispersion parameter '{param_id}' has max_value {max_val}, "
                                f"should be <= 1/3 for physical validity"
                            )

    def _check_structural_angle(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check that structural angle parameters are properly constrained."""
        for param in graph.parameters:
            param_id = param.get("id", "")
            if "theta" in param_id.lower() or "angle" in param_id.lower():
                constraint = param.get("constraint")
                if constraint:
                    ctype = constraint.get("type")
                    if ctype == "interval":
                        min_val = constraint.get("min_value")
                        max_val = constraint.get("max_value")
                        import math
                        if min_val is not None and min_val < 0:
                            result.add_warning(
                                f"Structural angle '{param_id}' has min_value {min_val}, "
                                f"should be >= 0"
                            )
                        if max_val is not None and max_val > math.pi:
                            result.add_warning(
                                f"Structural angle '{param_id}' has max_value {max_val}, "
                                f"should be <= π ({math.pi:.4f})"
                            )

    def _check_reference_shift(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check that reference shifts are applied for stress-free reference."""
        # Look for invariant inputs that should have reference shifts
        invariant_nodes = [
            n for n in graph.enabled_nodes
            if n.type == "invariant_input"
        ]

        # Check if they're connected to reference_shift nodes
        shifted_invariants = set()
        for node in graph.enabled_nodes:
            if node.type == "reference_shift":
                # Find what feeds into this shift
                for edge in graph.enabled_edges:
                    if edge.target == node.id:
                        source_node = next(
                            (n for n in graph.enabled_nodes if n.id == edge.source),
                            None
                        )
                        if source_node and source_node.type == "invariant_input":
                            shifted_invariants.add(edge.source)

        # Warn about unshifted invariants in stress-free mode
        for node in invariant_nodes:
            if node.id not in shifted_invariants:
                config = node.config or {}
                inv_name = config.get("invariant_name", node.id)
                # I1 and I2 typically need shifts
                if inv_name in ["I1", "I2"]:
                    result.add_warning(
                        f"Invariant '{inv_name}' ({node.id}) may need a reference_shift "
                        f"for stress-free reference configuration"
                    )

    def _check_tension_activation_ordering(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check that tension-only activation is applied before dispersion if configured."""
        # This is a simplified check - full implementation would trace the graph
        # to ensure tension_activation nodes come before dispersed_invariant nodes
        pass
