"""Example graph specifications for testing and documentation."""

from cann_graph.schema.graph import GraphSpec
from cann_graph.schema.nodes import NodeSpec
from cann_graph.schema.edges import EdgeSpec
from cann_graph.schema.enums import NodeCategory


def create_constant_energy_graph() -> GraphSpec:
    """Create a constant energy graph: ψ = c.
    
    This is the simplest possible energy model (Level A1).
    Expected stress: zero.
    """
    return GraphSpec(
        graph_id="constant_energy",
        nodes=[
            NodeSpec(
                id="constant",
                type="constant",
                category=NodeCategory.INPUT,
                config={"value": 1.0},
            ),
            NodeSpec(
                id="psi",
                type="energy_output",
                category=NodeCategory.OUTPUT,
            ),
        ],
        edges=[
            EdgeSpec(id="e1", source="constant", target="psi"),
        ],
        outputs=["psi"],
    )


def create_linear_i1_graph() -> GraphSpec:
    """Create a linear shifted isotropic energy graph: ψ = w * (I₁ - 3).
    
    This is a Neo-Hookean-like term (Level A2, B1).
    """
    return GraphSpec(
        graph_id="linear_i1",
        nodes=[
            NodeSpec(
                id="i1_input",
                type="invariant_input",
                category=NodeCategory.INVARIANT,
                config={"invariant_name": "I1"},
            ),
            NodeSpec(
                id="i1_shift",
                type="reference_shift",
                category=NodeCategory.INVARIANT,
                config={"reference": 3.0},
                parameter_refs=[],
            ),
            NodeSpec(
                id="weighted",
                type="weighted_sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 1},
                parameter_refs=["w_i1"],
            ),
            NodeSpec(
                id="psi",
                type="energy_output",
                category=NodeCategory.OUTPUT,
            ),
        ],
        edges=[
            EdgeSpec(id="e1", source="i1_input", target="i1_shift"),
            EdgeSpec(id="e2", source="i1_shift", target="weighted"),
            EdgeSpec(id="e3", source="weighted", target="psi"),
        ],
        outputs=["psi"],
    )


def create_quadratic_i1_graph() -> GraphSpec:
    """Create a quadratic energy graph: ψ = w * (I₁ - 3)².
    
    This tests autograd derivatives versus analytical (Level A3).
    """
    return GraphSpec(
        graph_id="quadratic_i1",
        nodes=[
            NodeSpec(
                id="i1_input",
                type="invariant_input",
                category=NodeCategory.INVARIANT,
                config={"invariant_name": "I1"},
            ),
            NodeSpec(
                id="i1_shift",
                type="reference_shift",
                category=NodeCategory.INVARIANT,
                config={"reference": 3.0},
            ),
            NodeSpec(
                id="square",
                type="square",
                category=NodeCategory.ENERGY,
            ),
            NodeSpec(
                id="weighted",
                type="weighted_sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 1},
                parameter_refs=["w_i1"],
            ),
            NodeSpec(
                id="psi",
                type="energy_output",
                category=NodeCategory.OUTPUT,
            ),
        ],
        edges=[
            EdgeSpec(id="e1", source="i1_input", target="i1_shift"),
            EdgeSpec(id="e2", source="i1_shift", target="square"),
            EdgeSpec(id="e3", source="square", target="weighted"),
            EdgeSpec(id="e4", source="weighted", target="psi"),
        ],
        outputs=["psi"],
    )


def create_mooney_rivlin_graph() -> GraphSpec:
    """Create a Mooney-Rivlin-like energy graph: ψ = c₁(I₁-3) + c₂(I₂-3).
    
    This tests multi-term isotropic energy (Level B2).
    """
    return GraphSpec(
        graph_id="mooney_rivlin",
        nodes=[
            # I1 branch
            NodeSpec(
                id="i1_input",
                type="invariant_input",
                category=NodeCategory.INVARIANT,
                config={"invariant_name": "I1"},
            ),
            NodeSpec(
                id="i1_shift",
                type="reference_shift",
                category=NodeCategory.INVARIANT,
                config={"reference": 3.0},
            ),
            NodeSpec(
                id="weighted_i1",
                type="weighted_sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 1},
                parameter_refs=["c1"],
            ),
            # I2 branch
            NodeSpec(
                id="i2_input",
                type="invariant_input",
                category=NodeCategory.INVARIANT,
                config={"invariant_name": "I2"},
            ),
            NodeSpec(
                id="i2_shift",
                type="reference_shift",
                category=NodeCategory.INVARIANT,
                config={"reference": 3.0},
            ),
            NodeSpec(
                id="weighted_i2",
                type="weighted_sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 1},
                parameter_refs=["c2"],
            ),
            # Sum
            NodeSpec(
                id="sum",
                type="sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 2},
            ),
            NodeSpec(
                id="psi",
                type="energy_output",
                category=NodeCategory.OUTPUT,
            ),
        ],
        edges=[
            # I1 path
            EdgeSpec(id="e1", source="i1_input", target="i1_shift"),
            EdgeSpec(id="e2", source="i1_shift", target="weighted_i1"),
            # I2 path
            EdgeSpec(id="e3", source="i2_input", target="i2_shift"),
            EdgeSpec(id="e4", source="i2_shift", target="weighted_i2"),
            # Sum
            EdgeSpec(id="e5", source="weighted_i1", target="sum"),
            EdgeSpec(id="e6", source="weighted_i2", target="sum"),
            EdgeSpec(id="e7", source="sum", target="psi"),
        ],
        outputs=["psi"],
    )


def create_four_basis_expansion_graph(invariant_name: str = "I1") -> GraphSpec:
    """Create a four-basis expansion for one invariant.
    
    Creates pathways for:
    - I (identity)
    - exp(I) - 1
    - I²
    - exp(I²) - 1
    
    All paths end at the same psi sum node.
    """
    shift_ref = 3.0 if invariant_name == "I1" or invariant_name == "I2" else 1.0
    
    return GraphSpec(
        graph_id=f"four_basis_{invariant_name.lower()}",
        nodes=[
            # Input
            NodeSpec(
                id=f"{invariant_name.lower()}_input",
                type="invariant_input",
                category=NodeCategory.INVARIANT,
                config={"invariant_name": invariant_name},
            ),
            NodeSpec(
                id=f"{invariant_name.lower()}_shift",
                type="reference_shift",
                category=NodeCategory.INVARIANT,
                config={"reference": shift_ref},
            ),
            # Identity path
            NodeSpec(
                id=f"{invariant_name.lower()}_identity",
                type="identity",
                category=NodeCategory.ENERGY,
                parameter_refs=["w_identity"],
            ),
            # exp(I) - 1 path
            NodeSpec(
                id=f"{invariant_name.lower()}_exp",
                type="exp_minus_one",
                category=NodeCategory.ENERGY,
                parameter_refs=["w1_exp", "w2_exp"],
            ),
            # I² path
            NodeSpec(
                id=f"{invariant_name.lower()}_square",
                type="square",
                category=NodeCategory.ENERGY,
            ),
            NodeSpec(
                id=f"{invariant_name.lower()}_square_weighted",
                type="weighted_sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 1},
                parameter_refs=["w_square"],
            ),
            # exp(I²) - 1 path
            NodeSpec(
                id=f"{invariant_name.lower()}_exp_square",
                type="exp_square_minus_one",
                category=NodeCategory.ENERGY,
                parameter_refs=["w1_exp_sq", "w2_exp_sq"],
            ),
            # Sum
            NodeSpec(
                id="sum",
                type="sum",
                category=NodeCategory.ENERGY,
                config={"num_inputs": 4},
            ),
            NodeSpec(
                id="psi",
                type="energy_output",
                category=NodeCategory.OUTPUT,
            ),
        ],
        edges=[
            # Common input
            EdgeSpec(id="e1", source=f"{invariant_name.lower()}_input", 
                     target=f"{invariant_name.lower()}_shift"),
            # Identity path
            EdgeSpec(id="e2", source=f"{invariant_name.lower()}_shift", 
                     target=f"{invariant_name.lower()}_identity"),
            # exp(I) - 1 path
            EdgeSpec(id="e3", source=f"{invariant_name.lower()}_shift", 
                     target=f"{invariant_name.lower()}_exp"),
            # I² path
            EdgeSpec(id="e4", source=f"{invariant_name.lower()}_shift", 
                     target=f"{invariant_name.lower()}_square"),
            EdgeSpec(id="e5", source=f"{invariant_name.lower()}_square", 
                     target=f"{invariant_name.lower()}_square_weighted"),
            # exp(I²) - 1 path
            EdgeSpec(id="e6", source=f"{invariant_name.lower()}_shift", 
                     target=f"{invariant_name.lower()}_exp_square"),
            # Sum
            EdgeSpec(id="e7", source=f"{invariant_name.lower()}_identity", target="sum"),
            EdgeSpec(id="e8", source=f"{invariant_name.lower()}_exp", target="sum"),
            EdgeSpec(id="e9", source=f"{invariant_name.lower()}_square_weighted", target="sum"),
            EdgeSpec(id="e10", source=f"{invariant_name.lower()}_exp_square", target="sum"),
            EdgeSpec(id="e11", source="sum", target="psi"),
        ],
        outputs=["psi"],
    )


EXAMPLE_GRAPHS = {
    "constant": create_constant_energy_graph,
    "linear_i1": create_linear_i1_graph,
    "quadratic_i1": create_quadratic_i1_graph,
    "mooney_rivlin": create_mooney_rivlin_graph,
    "four_basis_i1": lambda: create_four_basis_expansion_graph("I1"),
    "four_basis_i2": lambda: create_four_basis_expansion_graph("I2"),
}
