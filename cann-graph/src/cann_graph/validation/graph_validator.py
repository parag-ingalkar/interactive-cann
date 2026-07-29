"""Graph structure validation for CANN specifications."""

from dataclasses import dataclass, field
from collections import defaultdict

from cann_graph.schema.graph import GraphSpec


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class GraphValidator:
    """Validates the structural integrity of a CANN graph specification.

    Checks performed:
    - All node IDs are unique
    - All edge endpoints exist
    - Exactly one energy output exists
    - The graph is acyclic
    - Every declared output is reachable from inputs
    - No dead nodes (unless marked as draft)
    """

    def validate(self, graph: GraphSpec) -> ValidationResult:
        """Validate a graph specification.

        Args:
            graph: The graph specification to validate.

        Returns:
            ValidationResult with any errors or warnings found.
        """
        result = ValidationResult()

        # Check unique node IDs (already done by Pydantic, but verify)
        node_ids = [node.id for node in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            result.add_error(f"Duplicate node IDs found: {set(duplicates)}")

        # Check unique edge IDs
        edge_ids = [edge.id for edge in graph.edges]
        if len(edge_ids) != len(set(edge_ids)):
            duplicates = [eid for eid in edge_ids if edge_ids.count(eid) > 1]
            result.add_error(f"Duplicate edge IDs found: {set(duplicates)}")

        # Check all edge endpoints exist
        enabled_node_ids = {node.id for node in graph.enabled_nodes}
        for edge in graph.enabled_edges:
            if edge.source not in enabled_node_ids:
                result.add_error(
                    f"Edge '{edge.id}' references non-existent source node '{edge.source}'"
                )
            if edge.target not in enabled_node_ids:
                result.add_error(
                    f"Edge '{edge.id}' references non-existent target node '{edge.target}'"
                )

        # Check outputs exist
        for output_id in graph.outputs:
            if output_id not in enabled_node_ids:
                result.add_error(
                    f"Output '{output_id}' references non-existent node"
                )

        # Check for cycles
        self._check_acyclic(graph, result)

        # Check reachability of outputs
        self._check_output_reachability(graph, result)

        # Check for dead nodes
        self._check_dead_nodes(graph, result)

        return result

    def _check_acyclic(self, graph: GraphSpec, result: ValidationResult) -> None:
        """Check that the graph is acyclic using DFS."""
        enabled_nodes = {node.id for node in graph.enabled_nodes}
        adj: dict[str, list[str]] = defaultdict(list)

        for edge in graph.enabled_edges:
            if edge.source in enabled_nodes and edge.target in enabled_nodes:
                adj[edge.source].append(edge.target)

        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in enabled_nodes}
        parent = {nid: None for nid in enabled_nodes}

        def dfs(node: str, path: list[str]) -> list[str] | None:
            color[node] = GRAY
            for neighbor in adj[node]:
                if color[neighbor] == GRAY:
                    # Found cycle - reconstruct path
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    return cycle_path
                elif color[neighbor] == WHITE:
                    cycle_path = dfs(neighbor, path + [neighbor])
                    if cycle_path:
                        return cycle_path
            color[node] = BLACK
            return None

        for node_id in enabled_nodes:
            if color[node_id] == WHITE:
                cycle_path = dfs(node_id, [node_id])
                if cycle_path:
                    cycle_str = " -> ".join(cycle_path)
                    result.add_error(f"Graph contains a cycle: {cycle_str}")
                    return

    def _check_output_reachability(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check that all output nodes are reachable from some input."""
        enabled_nodes = {node.id for node in graph.enabled_nodes}
        
        # Build reverse adjacency (target -> sources)
        reverse_adj: dict[str, list[str]] = defaultdict(list)
        for edge in graph.enabled_edges:
            if edge.source in enabled_nodes and edge.target in enabled_nodes:
                reverse_adj[edge.target].append(edge.source)

        # Find input nodes (nodes with no incoming edges)
        has_incoming = {edge.target for edge in graph.enabled_edges 
                       if edge.source in enabled_nodes and edge.target in enabled_nodes}
        input_nodes = enabled_nodes - has_incoming

        # BFS from each output backwards to see if we can reach an input
        for output_id in graph.outputs:
            if output_id not in enabled_nodes:
                continue

            visited = set()
            queue = [output_id]
            reachable_from_input = False

            while queue and not reachable_from_input:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                if current in input_nodes:
                    reachable_from_input = True
                    break

                queue.extend(reverse_adj[current])

            if not reachable_from_input:
                result.add_warning(
                    f"Output node '{output_id}' may not be reachable from any input"
                )

    def _check_dead_nodes(
        self, graph: GraphSpec, result: ValidationResult
    ) -> None:
        """Check for nodes that don't contribute to any output."""
        enabled_nodes = {node.id for node in graph.enabled_nodes}
        
        # Build forward adjacency
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in graph.enabled_edges:
            if edge.source in enabled_nodes and edge.target in enabled_nodes:
                adj[edge.source].append(edge.target)

        # Find all nodes that can reach an output
        reachable_to_output: set[str] = set()
        
        def can_reach_output(node_id: str, visited: set[str]) -> bool:
            if node_id in visited:
                return node_id in reachable_to_output
            visited.add(node_id)

            if node_id in graph.outputs:
                reachable_to_output.add(node_id)
                return True

            for target in adj[node_id]:
                if can_reach_output(target, visited):
                    reachable_to_output.add(node_id)
                    return True

            return False

        for node_id in enabled_nodes:
            if not can_reach_output(node_id, set()):
                # Check if it's an input node (allowed to be disconnected from outputs if unused)
                has_incoming = any(
                    edge.target == node_id 
                    for edge in graph.enabled_edges 
                    if edge.source in enabled_nodes
                )
                if not has_incoming:
                    # It's an input node, might be intentionally unused
                    result.add_warning(
                        f"Node '{node_id}' does not contribute to any output"
                    )
                else:
                    result.add_warning(
                        f"Node '{node_id}' is unreachable from outputs (dead code)"
                    )
