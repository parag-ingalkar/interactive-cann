"""Parameter store for managing shared trainable variables in CANN models.

Note: TensorFlow is imported lazily to allow schema validation without TF installed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

from cann_graph.schema.parameters import ParameterSpec, ConstraintSpec
from cann_graph.schema.enums import ParameterConstraintType


class IntervalConstraint:
    """Interval constraint: clips values to [min_value, max_value]."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = tf.constant(min_value, dtype=tf.float64)
        self.max_value = tf.constant(max_value, dtype=tf.float64)

    def __call__(self, value: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(value, self.min_value, self.max_value)


class NonNegativeConstraint:
    """Non-negativity constraint: clips values to [0, ∞)."""

    def __call__(self, value: tf.Tensor) -> tf.Tensor:
        return tf.maximum(value, tf.zeros_like(value))


class PositiveConstraint:
    """Positivity constraint: ensures values are strictly positive."""

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = tf.constant(epsilon, dtype=tf.float64)

    def __call__(self, value: tf.Tensor) -> tf.Tensor:
        return tf.maximum(value, self.epsilon)


class UnitIntervalConstraint:
    """Unit interval constraint: clips values to [0, 1]."""

    def __call__(self, value: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(value, 0.0, 1.0)


def build_constraint(spec: ConstraintSpec):
    """Build a constraint function from a ConstraintSpec."""
    if spec.type == ParameterConstraintType.INTERVAL:
        return IntervalConstraint(spec.min_value, spec.max_value)
    elif spec.type == ParameterConstraintType.NON_NEGATIVE:
        return NonNegativeConstraint()
    elif spec.type == ParameterConstraintType.POSITIVE:
        return PositiveConstraint()
    elif spec.type == ParameterConstraintType.UNIT_INTERVAL:
        return UnitIntervalConstraint()
    else:
        raise ValueError(f"Unknown constraint type: {spec.type}")


class ParameterStore:
    """Manages all trainable and fixed parameters for a CANN model.

    The ParameterStore ensures that:
    - Each unique parameter_id maps to exactly one tf.Variable
    - Parameters are initialized once with specified initial values
    - Constraints are applied after optimizer updates
    - Trainable/fixed status is respected
    """

    def __init__(self):
        self._parameters: dict[str, tf.Variable] = {}
        self._constraints: dict[str, callable] = {}
        self._trainable: dict[str, bool] = {}
        self._specs: dict[str, ParameterSpec] = {}

    def get_or_create(self, spec: ParameterSpec) -> tf.Variable:
        """Get or create a parameter variable.

        Args:
            spec: The parameter specification.

        Returns:
            The tf.Variable for this parameter.

        Raises:
            ValueError: If trying to redefine an existing parameter with different properties.
        """
        param_id = spec.full_id

        if param_id in self._parameters:
            # Verify consistency
            existing_spec = self._specs[param_id]
            if (
                existing_spec.initial_value != spec.initial_value
                or existing_spec.trainable != spec.trainable
            ):
                raise ValueError(
                    f"Parameter '{param_id}' already exists with different properties. "
                    f"Existing: {existing_spec}, New: {spec}"
                )
            return self._parameters[param_id]

        # Create new variable
        initial_value = spec.initial_value
        if isinstance(initial_value, list):
            initial_value = tf.constant(initial_value, dtype=tf.float64)
        else:
            initial_value = tf.constant(initial_value, dtype=tf.float64)

        var = tf.Variable(
            initial_value,
            name=param_id.replace(".", "_"),
            dtype=tf.float64,
            trainable=spec.trainable,
        )

        self._parameters[param_id] = var
        self._trainable[param_id] = spec.trainable
        self._specs[param_id] = spec

        # Build constraint if specified
        if spec.constraint is not None:
            self._constraints[param_id] = build_constraint(spec.constraint)

        return var

    def get(self, parameter_id: str) -> tf.Variable:
        """Get a parameter variable by ID.

        Args:
            parameter_id: The fully qualified parameter ID.

        Returns:
            The tf.Variable for this parameter.

        Raises:
            KeyError: If the parameter does not exist.
        """
        if parameter_id not in self._parameters:
            available = ", ".join(sorted(self._parameters.keys()))
            raise KeyError(
                f"Parameter '{parameter_id}' not found. Available: {available}"
            )
        return self._parameters[parameter_id]

    def has(self, parameter_id: str) -> bool:
        """Check if a parameter exists."""
        return parameter_id in self._parameters

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        """Get all trainable variables."""
        return [
            var for param_id, var in self._parameters.items()
            if self._trainable.get(param_id, False)
        ]

    @property
    def all_variables(self) -> list[tf.Variable]:
        """Get all variables (trainable and fixed)."""
        return list(self._parameters.values())

    @property
    def parameter_dict(self) -> dict[str, tf.Tensor]:
        """Get a dictionary of parameter names to current values."""
        return {param_id: var.read_value() for param_id, var in self._parameters.items()}

    def apply_constraints(self) -> None:
        """Apply constraints to all constrained parameters.

        This should be called after each optimizer update to ensure
        parameters remain within their valid ranges.
        """
        for param_id, constraint in self._constraints.items():
            var = self._parameters[param_id]
            constrained_value = constraint(var.read_value())
            var.assign(constrained_value)

    def set_trainable(self, parameter_id: str, trainable: bool) -> None:
        """Set whether a parameter is trainable."""
        if parameter_id in self._parameters:
            self._parameters[parameter_id].trainable = trainable
            self._trainable[parameter_id] = trainable

    def clear(self) -> None:
        """Clear all parameters (for testing)."""
        self._parameters.clear()
        self._constraints.clear()
        self._trainable.clear()
        self._specs.clear()
