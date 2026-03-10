## 2026-03-09 - Initializing Bolt Journal
**Learning:** Starting the hunt for optimizations in a quantum entanglement simulation.
**Action:** Explore the codebase for O(n^2) operations, unnecessary re-calculations, or inefficient data structures.

## 2026-03-09 - Vectorized Lie Algebra and Power Series
**Learning:** Matrix commutator projections and recursive Lie series were identified as major bottlenecks using cProfile. Vectorizing these using `np.einsum` and pre-calculating flattened basis projections yielded significant speedups.
**Action:** Always check for repeated `np.trace` and matrix operations in loops. Use `np.einsum` for weighted summations of matrices.

## 2026-03-09 - Comprehensive Vectorization of Lie Evolution
**Learning:** The previous optimization was incomplete as the ODE solver and inner simulation loops still contained matrix-trace projections and manual summations. Fully vectorizing the Hamiltonian projection and generator summation using `np.einsum` and pre-calculated tensors achieved an order-of-magnitude performance gain in the evolution steps.
**Action:** Profile not just initialization, but also step-wise evolution functions. Vectorize all basis projections and matrix summations.
