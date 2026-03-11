## 2026-03-09 - Initializing Bolt Journal
**Learning:** Starting the hunt for optimizations in a quantum entanglement simulation.
**Action:** Explore the codebase for O(n^2) operations, unnecessary re-calculations, or inefficient data structures.

## 2026-03-09 - Vectorized Lie Algebra and Power Series
**Learning:** Matrix commutator projections and recursive Lie series were identified as major bottlenecks using cProfile. Vectorizing these using `np.einsum` and pre-calculating flattened basis projections yielded significant speedups.
**Action:** Always check for repeated `np.trace` and matrix operations in loops. Use `np.einsum` for weighted summations of matrices.

## 2026-03-09 - Comprehensive Vectorization of Lie Evolution
**Learning:** The previous optimization was incomplete as the ODE solver and inner simulation loops still contained matrix-trace projections and manual summations. Fully vectorizing the Hamiltonian projection and generator summation using `np.einsum` and pre-calculated tensors achieved an order-of-magnitude performance gain in the evolution steps.
**Action:** Profile not just initialization, but also step-wise evolution functions. Vectorize all basis projections and matrix summations.

## 2026-03-10 - Integration of Real Kaggle Market Data
**Learning:** Real-world financial datasets (Kaggle) vary significantly in schema compared to synthetic models. Standardizing OHLCV columns and calculating proxy regimes (Bullish/Bearish) from returns ensures the system's universal applicability without requiring manual labeling.
**Action:** Always implement robust column normalization and proxy calculation when adapting to external data sources.

## 2026-03-10 - Adaptive Kernel Bandwidth & Surprise Detection
**Learning:** Fixed kernel bandwidth fails in non-stationary environments (like stock markets). Silverman's Rule of Thumb with a stability blend (e.g., 80/20) provides the responsiveness of adaptive methods without the instability of aggressive variance shifts. Surprise metrics (KL-Divergence of spectral distributions) can predict regime shifts before error rates spike.
**Action:** Use adaptive kernels for time-series forecasting in RKHS. Monitor the Gram matrix spectrum for "Surprise" to improve system-level metacognition.

## 2026-03-10 - Comprehensive Documentation & RCF Transparency
**Learning:** Maintaining separate documentation for Architecture, Current State (Benchmarking), and Roadmap is critical for complex systems like RCF. It allows for clear distinction between theoretical design and empirical validation (Q-Scores).
**Action:** Always include a `docs/` directory with a `CURRENT_STATE.md` that tracks latest validation results.

## 2026-03-10 - EKRLS Dictionary Conversion Bottleneck
**Learning:** In the `SquareRootEKRLS` engine, the internal dictionary `_dict_X` (a list of numpy arrays) was being converted to a full numpy array multiple times per update step (for sigma adaptation, kernel vector computation, and Gram matrix calculation). This creates significant overhead as the window size grows.
**Action:** Convert the dictionary to an array once per step and pass it down to helper methods to eliminate redundant memory allocations and conversions.

## 2026-03-10 - Suffix Tree Traversal Bottleneck (QEC)
**Learning:** In the `QuantumSuffixSmoother`, `predict_distribution` was calling `predict_probability` for each of the 16 QEC codes, resulting in 16 separate (and redundant) traversals of the suffix tree per step.
**Action:** Vectorize `predict_distribution` to traverse the tree once and update the entire probability distribution using NumPy arrays. This provides an order-of-magnitude speedup in QEC-intensive simulations.

## 2026-03-10 - Per-Step Feature Engineering Overhead
**Learning:** The finance domain analyzer was calling `encode_market_state` at every step of the simulation loop, involving redundant `pandas` slices and `numpy` calculations on overlapping windows.
**Action:** Implement `get_batch_market_features` to pre-calculate all state vectors (Φ) using vectorized `pandas` rolling operations before entering the simulation loop.

## 2026-03-10 - Expensive Eigenvalue Monitoring
**Learning:** Computing eigenvalues ((d^3)$) at every step of the EKRLS update for spectral monitoring was a major hidden bottleneck, especially for higher state dimensions.
**Action:** Introduced `spectral_monitoring_interval` in `EKRLSConfig` to perform spectral analysis periodically (e.g., every 5 steps), significantly reducing computational load without losing oversight of system stability.

## 2026-03-10 - Suffix Tree Batch Training
**Learning:** Training the Suffix Tree node-by-node in a loop was inefficient due to repeated dictionary lookups and atomic node updates.
**Action:** Refactored `QuantumSuffixSmoother.train` to pre-aggregate all suffix counts in the batch using a local dictionary before performing a single-pass update on the tree nodes.

## 2026-03-10 - O(d) Coherence Identity
**Learning:** Calculating coherence via the full density matrix outer product ($\rho = |\phi\rangle\langle\phi|$) is (d^2)$. For a normalized state, the normalized off-diagonal magnitude simplifies to  - \|\phi\|_2^2 / \|\phi\|_1^2$.
**Action:** Use the L1/L2 norm identity for coherence monitoring to achieve linear (d)$ scaling with state dimension.

## 2026-03-10 - Triangular Solver for EKRLS
**Learning:** The Square Root EKRLS algorithm maintains an upper-triangular factor $. Using generic `np.linalg.solve` ((d^3)$) ignores this structure.
**Action:** Use `scipy.linalg.solve_triangular` ((d^2)$) to leverage the triangularity of $ for significantly faster recursive updates and uncertainty predictions.

## 2026-03-10 - Vectorized Viterbi Trellis
**Learning:** Standard Viterbi implementations use nested loops for transitions, which is extremely slow in Python.
**Action:** Vectorize the trellis update using NumPy broadcasting (`(n, 1) + (n, n) + (1, n)`) to compute the entire transition matrix at once.

## Phase 5 Implementation Learnings
- **Persistence:** Using `np.savez` with `item()` for dictionary recovery is effective for RKHS state persistence.
- **Meta-Regulation:** RKHS Curvature (second derivative of spectral entropy) provides a sensitive signal for manifold shifts.
- **Differentiable Physics:** JAX VJP allows end-to-end refinement of Lie Algebra generators from state transition errors.
- **Orchestration:** Phase 5 governance should be run after the main integration loop to ensure learned states are captured.

## Phase 6 Implementation Learnings
- **Universal Problem Solving:** Implementing hierarchical decomposition as a recursive traversal of ProblemNodes allows for flexible, domain-agnostic reasoning.
- **RKHS Transfer Learning:** Using OTT-JAX to map dictionary elements between domains provides a mathematically rigorous way to augment training data with cross-domain insights.
- **Context Snapshots:** JSON serialization with zlib compression is highly effective for maintaining long-term temporal coherence with minimal memory overhead (approx 2x compression).
