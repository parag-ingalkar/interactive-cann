# CANN Graph

A Python package that compiles a validated neuron-level graph schema into a shared, multi-protocol Constitutive Artificial Neural Network (CANN) for biomechanical material modeling.

## Overview

This package provides:

1. **Graph Schema**: A declarative, versioned specification for constitutive model graphs using Pydantic v2.
2. **Validation**: Multi-layer validation (structural, type, parameter-sharing, physics) before compilation.
3. **Compiler**: Transforms validated graphs into differentiable TensorFlow/Keras models.
4. **Multi-Protocol Support**: Nine protocol-specific kinematic branches sharing one material law.
5. **Stress Computation**: Autograd-based stress prediction via differentiation of strain energy.
6. **Symbolic Export**: Discovery and export of closed-form constitutive equations.

## Installation

```bash
pip install cann_graph
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from cann_graph.schema import GraphSpec, NodeSpec, EdgeSpec, ParameterSpec
from cann_graph.compiler import GraphCompiler, ParameterStore

# Define a simple linear energy graph: ψ = w * (I₁ - 3)
graph = GraphSpec(
    graph_id="linear_i1",
    nodes=[
        NodeSpec(id="i1_shift", type="reference_shift", config={"reference": 3.0}),
        NodeSpec(id="weighted", type="weighted_sum", config={"weights": [1.0]}),
        NodeSpec(id="psi", type="energy_output"),
    ],
    edges=[
        EdgeSpec(id="e1", source="i1_shift", target="weighted"),
        EdgeSpec(id="e2", source="weighted", target="psi"),
    ],
    outputs=["psi"],
)

# Compile to TensorFlow model
compiler = GraphCompiler()
compiled = compiler.compile(graph)
```

## Verification Ladder

The package follows a progressive verification approach:

- **Level A**: Pure graph compiler (constant, linear, quadratic energy)
- **Level B**: Standard constitutive forms (Neo-Hookean, Mooney-Rivlin)
- **Level C**: Aligned anisotropy (fiber-reinforced models)
- **Level D**: Dispersion-aware CANNs (trainable κ)
- **Level E**: Full nine-protocol thesis model

See `examples/` for executable specifications at each level.

## Architecture

```
src/cann_graph/
├── schema/         # Pydantic models for graph specification
├── validation/     # Multi-layer validation (structure, types, physics)
├── operations/     # Typed atomic operations with TF + symbolic implementations
├── compiler/       # Graph compiler, parameter store, protocol branches
├── training/       # Losses, trainer, multi-start orchestration
├── data/           # Protocol datasets, synthetic data generation
└── reporting/      # Run manifests, equation reports, result bundles
```

## Key Design Decisions

1. **Three-domain separation**: Protocol, physics, and energy graph schemas are distinct but related.
2. **Typed operations**: No arbitrary code nodes; every operation has a registered type with explicit signatures.
3. **Explicit parameter sharing**: Reusing a `parameter_id` means reusing the exact same `tf.Variable`.
4. **float64 by default**: Constitutive-model verification requires double precision.
5. **Constraint enforcement**: Interval and non-negativity constraints applied after every optimizer step.

## Development Status

This is a core-first implementation. The following are intentionally **out of scope** for v1:

- React Flow canvas / UI
- FastAPI endpoints
- User authentication, persistence, job queues
- Arbitrary Python expressions from users
- Finite-element solver integration
- Direct CSV/Excel UX

These will be added once the core compiler is proven trustworthy.

## License

MIT
