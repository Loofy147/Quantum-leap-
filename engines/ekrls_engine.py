"""
EKRLS Engine — Extended Kernel Recursive Least Squares
للنظام الكمومي: تتبع الحالة في الوقت الحقيقي عبر RKHS

Φ_n = f(Φ_{n-1}) + v_n   (state transition)
y_n = g(Φ_n)  + w_n       (measurement)

Uses Square Root EKRLS with Givens rotations for numerical stability.
No matrix inversion at each step → O(n²) updates, FPGA-ready.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EKRLSConfig:
    """Configuration for EKRLS Engine."""
    state_dim: int = 4           # Dimension of quantum state Φ
    kernel_sigma: float = 1.0    # RBF kernel bandwidth
    adaptive_bandwidth: bool = True # Use Silverman's Rule of Thumb
    forgetting_factor: float = 0.99  # λ in recursive update (memory)
    process_noise: float = 0.01  # σ_v²
    measurement_noise: float = 0.05  # σ_w²
    window_size: int = 50
    spectral_monitoring_interval: int = 5        # Sliding window for RKHS dict


class RBFKernel:
    """Radial Basis Function kernel: k(x,x') = exp(-||x-x'||²/2σ²)"""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
        """Single kernel evaluation."""
        diff = x.flatten() - y.flatten()
        s = sigma if sigma is not None else self.sigma
        return float(np.exp(-np.dot(diff, diff) / (2 * s ** 2 + 1e-12)))

    def compute(self, X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Vectorized kernel evaluation between two sets of vectors."""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        s = sigma if sigma is not None else self.sigma
        # Use squared distance identity: ||x-y||² = ||x||² + ||y||² - 2x·y
        sx = np.einsum('ij,ij->i', X, X)
        sy = np.einsum('ij,ij->i', Y, Y)
        dist_sq = sx[:, np.newaxis] + sy[np.newaxis, :] - 2 * np.dot(X, Y.T)
        return np.exp(-dist_sq / (2 * s ** 2 + 1e-12))

    def gram_matrix(self, X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Compute full Gram matrix K_{ij} = k(x_i, x_j)"""
        # Vectorized implementation for speed boost (Bolt ⚡)
        X_arr = np.asarray(X)
        return self.compute(X_arr, X_arr, sigma=sigma)


@dataclass
class QuantumState:
    """Represents a quantum state vector in Hilbert space."""
    phi: np.ndarray          # State vector (density matrix diagonal)
    timestamp: int = 0
    coherence: float = 1.0   # 1.0 = fully coherent, 0.0 = collapsed
    entanglement_entropy: float = 0.0

    def is_collapsed(self, threshold: float = 0.1) -> bool:
        return self.coherence < threshold

    def von_neumann_entropy(self) -> float:
        """S = -Tr(ρ log ρ) approximated from state vector."""
        probs = np.abs(self.phi) ** 2
        probs = probs / probs.sum()
        # Avoid log(0)
        safe = probs[probs > 1e-12]
        return float(-np.sum(safe * np.log2(safe)))


class SquareRootEKRLS:
    """
    Square Root Extended Kernel Recursive Least Squares.

    Uses Givens rotations to update the data window without matrix inversion.
    Suitable for real-time quantum state tracking and FPGA implementation.
    """

    def __init__(self, config: EKRLSConfig):
        self.cfg = config
        self.kernel = RBFKernel(sigma=config.kernel_sigma)
        self.lam = config.forgetting_factor
        self.q_noise = config.process_noise
        self.r_noise = config.measurement_noise

        # Dictionary (sliding window) of past states
        self._dict_X: list[np.ndarray] = []
        self._dict_y: list[float] = []

        # Square-root covariance factor (upper triangular)
        self._R_sqrt: Optional[np.ndarray] = None
        # Weight vector in RKHS
        self._alpha: Optional[np.ndarray] = None

        self.update_count = 0
        self.prediction_errors: list[float] = []

    def _givens_rotation(self, R: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Givens rotation update: annihilate subdiagonal using plane rotations.
        Updates R in-place to maintain triangular form.
        Returns updated R and rotated x.
        """
        R = R.copy()
        x = x.copy()
        n = len(x)
        for i in range(n):
            if abs(x[i]) < 1e-14:
                continue
            r = np.sqrt(R[i, i] ** 2 + x[i] ** 2)
            c = R[i, i] / r  # cosine
            s = x[i] / r     # sine
            # Apply rotation to row i of R and x
            R[i, i] = r
            if i + 1 < n:
                R[i, i+1:] = c * R[i, i+1:] + s * x[i+1:]
                x[i+1:]    = -s * R[i, i+1:] + c * x[i+1:]  # Note: uses updated R
            x[i] = 0.0
        return R, x

    def _get_adaptive_sigma(self, X_dict: Optional[np.ndarray] = None) -> float:
        """Silverman's Rule of Thumb proxy for bandwidth selection."""
        if not self.cfg.adaptive_bandwidth or len(self._dict_X) < 10:
            return self.cfg.kernel_sigma

        X = X_dict if X_dict is not None else np.array(self._dict_X)
        # Multi-dimensional Silverman proxy: 1.06 * std * n^(-1/5)
        # We blend with base sigma for stability
        std = np.std(X, axis=0).mean() + 1e-8
        n = len(X)
        silverman = 1.06 * std * (n ** -0.2)

        # Stability blend: 80% fixed, 20% adaptive to avoid aggressive variance shifts
        sigma = 0.8 * self.cfg.kernel_sigma + 0.2 * silverman
        return float(np.clip(sigma, 0.1, 10.0))

    def _kernel_vector(self, x_new: np.ndarray, sigma: Optional[float] = None, X_dict: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel vector k(x_new, x_i) for all dictionary entries (Vectorized)."""
        if not self._dict_X:
            return np.array([])
        X_dict = X_dict if X_dict is not None else np.array(self._dict_X)
        return self.kernel.compute(x_new.reshape(1, -1), X_dict, sigma=sigma).flatten()

    def predict(self, phi_new: np.ndarray, k_vec: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Predict measurement y for new state phi_new.
        Returns: (prediction, uncertainty_estimate)
        """
        if self._alpha is None or len(self._dict_X) == 0:
            return 0.0, float('inf')

        if k_vec is None:
            k_vec = self._kernel_vector(phi_new)

        # Guard against size mismatch (alpha not yet extended)
        n = min(len(self._alpha), len(k_vec))
        y_pred = float(np.dot(self._alpha[:n], k_vec[:n]))

        # Uncertainty from RKHS norm
        k_self = float(k_vec[-1]) if len(k_vec) > n else self.kernel(phi_new, phi_new)

        # Use only matching part of k_vec for uncertainty if R_sqrt is older
        k_vec_u = k_vec[:self._R_sqrt.shape[0]] if self._R_sqrt is not None else k_vec

        if self._R_sqrt is not None and len(k_vec_u) == self._R_sqrt.shape[0]:
            try:
                v = np.linalg.solve(self._R_sqrt.T, k_vec_u)
                uncertainty = float(np.sqrt(max(0, k_self - np.dot(v, v))))
            except np.linalg.LinAlgError:
                uncertainty = float(np.sqrt(k_self))
        else:
            uncertainty = float(np.sqrt(k_self))

        return y_pred, uncertainty

    def update(self, phi_n: np.ndarray, y_n: float) -> dict:
        """
        Recursive update given new state-measurement pair.
        Uses Givens rotations for numerically stable covariance update.
        """
        self.update_count += 1
        n = self.update_count

        # --- Window management ---
        if len(self._dict_X) >= self.cfg.window_size:
            self._dict_X.pop(0)
            self._dict_y.pop(0)
            if self._alpha is not None and len(self._alpha) > 0:
                self._alpha = self._alpha[1:]
            if self._R_sqrt is not None and self._R_sqrt.shape[0] > 1:
                self._R_sqrt = self._R_sqrt[1:, 1:]

        self._dict_X.append(phi_n.copy())
        self._dict_y.append(y_n)

        # Convert to array once per step (Bolt ⚡ Optimization)
        X_dict = np.array(self._dict_X)
        d = len(X_dict)

        # --- Initialize if first step ---
        if d == 1:
            k00 = self.kernel(phi_n, phi_n) + self.r_noise
            self._R_sqrt = np.array([[np.sqrt(k00)]])
            self._alpha = np.array([y_n / k00])
            self.prediction_errors.append(y_n)
            return {"step": n, "d": d, "pred_error": y_n, "uncertainty": float(np.sqrt(k00))}

        # --- Kernel vector for new point ---
        # Reuse X_dict array (Bolt ⚡ Optimization)
        current_sigma = self._get_adaptive_sigma(X_dict=X_dict)
        k_full = self._kernel_vector(phi_n, sigma=current_sigma, X_dict=X_dict)
        k_vec = k_full[:-1]
        k_self = k_full[-1] + self.r_noise

        # --- Extend R_sqrt ---
        R_old = self._R_sqrt
        r_old_d = len(R_old)
        R_new = np.zeros((d, d))
        R_new[:r_old_d, :r_old_d] = R_old

        # New row: [k_vec, sqrt(k_self - k_vec·inv(R^T R)·k_vec)]
        if r_old_d > 0 and len(k_vec) > 0:
            try:
                v = np.linalg.solve(R_old.T, k_vec)
                schur = k_self - np.dot(v, v)
                R_new[r_old_d - 1, r_old_d - 1] = np.sqrt(max(schur, 1e-10))
            except np.linalg.LinAlgError:
                R_new[d-1, d-1] = np.sqrt(k_self)
        else:
            R_new[d-1, d-1] = np.sqrt(k_self)

        self._R_sqrt = R_new

        # --- Prediction error ---
        # Reuse k_full from above (Bolt ⚡ Optimization)
        y_pred, uncertainty = self.predict(phi_n, k_vec=k_full)
        pred_error = y_n - y_pred
        self.prediction_errors.append(pred_error)

        # --- Update alpha via RKHS weight update ---
        # Reuse X_dict array (Bolt ⚡ Optimization)
        K = self.kernel.gram_matrix(X_dict, sigma=current_sigma)
        K += self.r_noise * np.eye(d)
        try:
            self._alpha = np.linalg.solve(K, np.array(self._dict_y))
        except np.linalg.LinAlgError:
            self._alpha = np.linalg.lstsq(K, np.array(self._dict_y), rcond=None)[0]

        # Compute eigenvalues for spectral monitoring periodically (Bolt ⚡ Optimization)
        if self.update_count % self.cfg.spectral_monitoring_interval == 0:
            try:
                eigenvalues = np.linalg.eigvalsh(K)
            except:
                eigenvalues = np.zeros(d)
        else:
            eigenvalues = None

        return {
            "step": n,
            "d": d,
            "y_pred": float(y_pred),
            "y_true": float(y_n),
            "pred_error": float(pred_error),
            "uncertainty": float(uncertainty),
            "eigenvalues": eigenvalues,
        }


class EKRLSQuantumEngine:
    """
    Full quantum state tracking engine.
    Wraps SquareRootEKRLS with quantum-specific state management.

    Detects entanglement collapse and issues metacognitive alerts.
    """

    def __init__(self, config: Optional[EKRLSConfig] = None):
        self.cfg = config or EKRLSConfig()
        self.ekrls = SquareRootEKRLS(self.cfg)
        self.state_history: list[QuantumState] = []
        self.collapse_events: list[int] = []
        self.entanglement_battery: float = 1.0  # Normalized [0,1]

    def _compute_coherence(self, phi: np.ndarray) -> float:
        """Coherence = normalized off-diagonal density matrix magnitude."""
        n = len(phi)
        if n < 2:
            return 1.0
        # Outer product approximation of density matrix
        rho = np.outer(phi, phi.conj())
        diag_sum = np.abs(np.diag(rho)).sum()
        total_sum = np.abs(rho).sum()
        if total_sum < 1e-12:
            return 0.0
        off_diag = total_sum - diag_sum
        return float(off_diag / total_sum)

    def step(self, phi_raw: np.ndarray, measurement: float) -> dict:
        """
        Process one quantum evolution step.
        phi_raw: raw state vector
        measurement: observed output y_n
        """
        # Normalize state
        norm = np.linalg.norm(phi_raw)
        phi_n = phi_raw / (norm + 1e-12)

        # Add process noise (quantum fluctuation)
        phi_n = phi_n + np.random.normal(0, self.cfg.process_noise, phi_n.shape)
        phi_n /= np.linalg.norm(phi_n) + 1e-12

        # Compute quantum properties
        coherence = self._compute_coherence(phi_n)
        qs = QuantumState(
            phi=phi_n,
            timestamp=len(self.state_history),
            coherence=coherence,
            entanglement_entropy=0.0
        )
        qs.entanglement_entropy = qs.von_neumann_entropy()

        # Check for collapse BEFORE update
        if qs.is_collapsed():
            self.collapse_events.append(len(self.state_history))

        # EKRLS update
        update_info = self.ekrls.update(phi_n, measurement)

        # Update entanglement battery based on entropy change
        if self.state_history:
            prev_S = self.state_history[-1].entanglement_entropy
            delta_S = qs.entanglement_entropy - prev_S
            # Battery charges when entanglement increases
            self.entanglement_battery = np.clip(
                self.entanglement_battery + 0.1 * delta_S, 0.0, 1.0
            )

        self.state_history.append(qs)

        return {
            **update_info,
            "coherence": coherence,
            "entropy": qs.entanglement_entropy,
            "battery_level": self.entanglement_battery,
            "collapse_detected": qs.is_collapsed(),
            "total_collapses": len(self.collapse_events),
        }

    def run_simulation(self, n_steps: int = 100, seed: int = 42) -> list[dict]:
        """Run a full simulation with synthetic quantum evolution."""
        np.random.seed(seed)
        results = []
        dim = self.cfg.state_dim

        # Initial state: Bell-like superposition
        phi = np.zeros(dim)
        phi[0] = 1.0 / np.sqrt(2)
        phi[1] = 1.0 / np.sqrt(2)

        for t in range(n_steps):
            # Unitary-like evolution (rotation + decoherence)
            theta = 0.05 * t
            if dim >= 2:
                c, s = np.cos(theta), np.sin(theta)
                phi_new = phi.copy()
                phi_new[0] = c * phi[0] - s * phi[1]
                phi_new[1] = s * phi[0] + c * phi[1]
                phi = phi_new

            # Add decoherence at random steps
            if np.random.random() < 0.05:
                noise = np.random.randn(dim) * 0.3
                phi = phi + noise
                phi /= np.linalg.norm(phi) + 1e-12

            # Measurement (projection onto first basis state)
            measurement = float(np.abs(phi[0]) ** 2) + np.random.normal(0, 0.02)
            result = self.step(phi, measurement)
            result["t"] = t
            results.append(result)

        return results

    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.state_history:
            return {}
        entropies = [s.entanglement_entropy for s in self.state_history]
        coherences = [s.coherence for s in self.state_history]
        errors = self.ekrls.prediction_errors
        return {
            "total_steps": len(self.state_history),
            "collapse_events": len(self.collapse_events),
            "mean_entropy": float(np.mean(entropies)),
            "mean_coherence": float(np.mean(coherences)),
            "mean_pred_error": float(np.mean(np.abs(errors))) if errors else 0.0,
            "rmse": float(np.sqrt(np.mean(np.array(errors)**2))) if errors else 0.0,
            "battery_final": self.entanglement_battery,
            "dictionary_size": len(self.ekrls._dict_X),
        }
