"""Operation registry for CANN graph."""

from typing import Type

from cann_graph.operations.base import Operation


class OperationRegistry:
    """Registry for all available CANN graph operations.

    Operations are registered by their type_name and can be retrieved
    for building compiled graphs.
    """

    _operations: dict[str, Type[Operation]] = {}

    @classmethod
    def register(cls, operation_class: Type[Operation]) -> Type[Operation]:
        """Register an operation class.

        Args:
            operation_class: The operation class to register.

        Returns:
            The same operation class (for decorator usage).

        Raises:
            ValueError: If an operation with the same type_name is already registered.
        """
        if operation_class.type_name in cls._operations:
            raise ValueError(
                f"Operation '{operation_class.type_name}' is already registered"
            )
        cls._operations[operation_class.type_name] = operation_class
        return operation_class

    @classmethod
    def get(cls, type_name: str) -> Type[Operation]:
        """Get an operation class by type name.

        Args:
            type_name: The operation type name.

        Returns:
            The operation class.

        Raises:
            KeyError: If the operation is not registered.
        """
        if type_name not in cls._operations:
            available = ", ".join(sorted(cls._operations.keys()))
            raise KeyError(
                f"Operation '{type_name}' not found. Available: {available}"
            )
        return cls._operations[type_name]

    @classmethod
    def list_operations(cls) -> list[str]:
        """List all registered operation type names."""
        return sorted(cls._operations.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered operations (for testing)."""
        cls._operations.clear()


# Convenience functions
def register_operation(operation_class: Type[Operation]) -> Type[Operation]:
    """Register an operation (convenience function)."""
    return OperationRegistry.register(operation_class)


def get_operation(type_name: str) -> Type[Operation]:
    """Get an operation by type name (convenience function)."""
    return OperationRegistry.get(type_name)
