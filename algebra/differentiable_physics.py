import jax
import jax.numpy as jnp
from typing import List, Tuple

def transition_error(generators: jnp.ndarray, g: jnp.ndarray, phi_t: jnp.ndarray, phi_next: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the squared error of the state transition.
    Φ_next ≈ exp(Σ g_i X_i) Φ_t
    For infinitesimal step: Φ_next - Φ_t ≈ (Σ g_i X_i) Φ_t
    """
    # L = Σ g_i X_i
    L = jnp.einsum('i,ijk->jk', g, generators)

    # Infinitesimal approximation
    pred_delta = jnp.matmul(L, phi_t)
    actual_delta = phi_next - phi_t

    return jnp.sum(jnp.abs(pred_delta - actual_delta)**2)

def compute_vjp_update(generators: List[jnp.ndarray], g: jnp.ndarray, phi_t: jnp.ndarray, phi_next: jnp.ndarray) -> jnp.ndarray:
    """
    Tier 2026: Recursive Self-Correction via Vector-Jacobian Product.
    Calculates the gradient of the transition error with respect to generators.
    """
    gens_jnp = jnp.array(generators)

    # Ensure inputs are JAX-compatible and real-valued for gradients if needed,
    # but Lie generators are often complex. JAX handles complex grad if we are careful.

    loss_val, vjp_fun = jax.vjp(lambda gens: transition_error(gens, g, phi_t, phi_next), gens_jnp)

    # We want to minimize the loss, so we take the gradient (VJP with 1.0)
    grads = vjp_fun(1.0)[0]

    return grads
