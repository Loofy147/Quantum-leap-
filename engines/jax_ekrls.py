import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple

class EKRLSState(NamedTuple):
    alpha: jnp.ndarray
    R: jnp.ndarray
    X_dict: jnp.ndarray
    step: int

@jax.jit
def jax_rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """JAX-accelerated RBF Kernel calculation."""
    # (x - y)^2 = x^2 + y^2 - 2xy
    sq_dist = jnp.sum(x**2, axis=1, keepdims=True) + jnp.sum(y**2, axis=1) - 2 * jnp.dot(x, y.T)
    return jnp.exp(-sq_dist / (2 * sigma**2))

@jax.jit
def ekrls_predict(state: EKRLSState, phi_new: jnp.ndarray, sigma: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-accelerated EKRLS Prediction."""
    if state.X_dict.shape[0] == 0:
        return jnp.array(0.0), jnp.array(1.0)

    k_vec = jax_rbf_kernel(phi_new.reshape(1, -1), state.X_dict, sigma).flatten()
    y_pred = jnp.dot(k_vec, state.alpha)

    # Uncertainty: 1 - k^T * (R^T * R)^-1 * k
    a = jax.lax.linalg.triangular_solve(state.R, k_vec.reshape(-1, 1), left_side=True, lower=False, transpose_a=True)
    uncertainty = 1.0 - jnp.sum(a**2)

    return y_pred, uncertainty

@jax.jit
def jax_von_neumann_entropy(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Tier 2026: GPU-Accelerated Von Neumann Entropy.
    S = -Tr(ρ log ρ) approximated from the state vector Φ.
    """
    probs = jnp.abs(phi)**2
    # Normalize to ensure valid probability distribution
    probs = probs / (jnp.sum(probs) + 1e-12)

    # Entropy calculation with safe log
    entropy = -jnp.sum(probs * jnp.log2(probs + 1e-12))
    return entropy
