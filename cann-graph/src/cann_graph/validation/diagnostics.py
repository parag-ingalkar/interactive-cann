"""Diagnostics module for validation."""

from dataclasses import dataclass, field

from cann_graph.schema.graph import GraphSpec
from cann_graph.validation.graph_validator import ValidationResult, GraphValidator
from cann_graph.validation.type_validator import TypeValidator
from cann_graph.validation.physics_validator import PhysicsValidator
from cann_graph.validation.sharing_validator import SharingValidator


@dataclass
class ValidationReport:
    """Complete validation report for a graph specification."""

    graph_id: str
    is_valid: bool = True
    structural_result: ValidationResult | None = None
    type_result: ValidationResult | None = None
    physics_result: ValidationResult | None = None
    sharing_result: ValidationResult | None = None
    
    @property
    def all_errors(self) -> list[str]:
        """Collect all errors from all validation stages."""
        errors = []
        if self.structural_result:
            errors.extend(self.structural_result.errors)
        if self.type_result:
            errors.extend(self.type_result.errors)
        if self.physics_result:
            errors.extend(self.physics_result.errors)
        if self.sharing_result:
            errors.extend(self.sharing_result.errors)
        return errors
    
    @property
    def all_warnings(self) -> list[str]:
        """Collect all warnings from all validation stages."""
        warnings = []
        if self.structural_result:
            warnings.extend(self.structural_result.warnings)
        if self.type_result:
            warnings.extend(self.type_result.warnings)
        if self.physics_result:
            warnings.extend(self.physics_result.warnings)
        if self.sharing_result:
            warnings.extend(self.sharing_result.warnings)
        return warnings


class ComprehensiveValidator:
    """Runs all validation checks on a graph specification."""

    def __init__(
        self,
        run_structural: bool = True,
        run_type: bool = True,
        run_physics: bool = True,
        run_sharing: bool = True,
        physics_mode_kwargs: dict | None = None,
    ):
        self.run_structural = run_structural
        self.run_type = run_type
        self.run_physics = run_physics
        self.run_sharing = run_sharing
        self.physics_mode_kwargs = physics_mode_kwargs or {}

    def validate(self, graph: GraphSpec) -> ValidationReport:
        """Run all validation checks on a graph.

        Args:
            graph: The graph specification to validate.

        Returns:
            ValidationReport with results from all validation stages.
        """
        report = ValidationReport(graph_id=graph.graph_id)

        # Structural validation
        if self.run_structural:
            validator = GraphValidator()
            report.structural_result = validator.validate(graph)
            if not report.structural_result.is_valid:
                report.is_valid = False

        # Type validation
        if self.run_type:
            validator = TypeValidator()
            report.type_result = validator.validate(graph)
            if not report.type_result.is_valid:
                report.is_valid = False

        # Physics validation
        if self.run_physics:
            validator = PhysicsValidator(**self.physics_mode_kwargs)
            report.physics_result = validator.validate(graph)
            if not report.physics_result.is_valid:
                report.is_valid = False

        # Sharing validation
        if self.run_sharing:
            validator = SharingValidator()
            report.sharing_result = validator.validate(graph)
            if not report.sharing_result.is_valid:
                report.is_valid = False

        return report


def validate_graph(
    graph: GraphSpec,
    strict: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """Convenience function to validate a graph.

    Args:
        graph: The graph specification to validate.
        strict: If True, warnings are treated as errors.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    validator = ComprehensiveValidator()
    report = validator.validate(graph)

    errors = report.all_errors
    warnings = report.all_warnings

    if strict and warnings:
        errors = errors + warnings
        warnings = []

    return len(errors) == 0, errors, warnings
