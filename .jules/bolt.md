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
