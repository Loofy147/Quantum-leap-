import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple, List

class StructuralLearner:
    """
    Phase 3 (Tier 2026): Bayesian Structural Learning with Optax
    Uses second-order optimization for faster structural refinement.
    """
    def __init__(self, n_gens: int, dim: int, learning_rate: float = 0.01):
        self.n_gens = n_gens
        self.dim = dim
        self.optimizer = optax.adamw(learning_rate=learning_rate)
        self.opt_state = None
        self.params = None

    def init_params(self, initial_generators: List[jnp.ndarray]):
        """Initialize parameters for optimization."""
        # Using real and imag parts separately to avoid complex grad issues
        self.params = jnp.array([g.flatten() for g in initial_generators])
        # We only refine the real parts for stability unless holomorphic is needed
        self.params = jnp.real(self.params)
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    def loss_fn(params: jnp.ndarray, state_history: jnp.ndarray, g: jnp.ndarray, n_gens: int, dim: int) -> jnp.ndarray:
        gens = params.reshape(n_gens, dim, dim)
        L = jnp.einsum('i,ijk->jk', g, gens)
        phi_t = state_history[:-1]
        phi_next = state_history[1:]
        # Ensure phi_t is real for calculation
        phi_t_real = jnp.real(phi_t[:, :dim])
        pred_delta = jnp.einsum('ij,tj->ti', L, phi_t_real)
        actual_delta = jnp.real(phi_next[:, :dim] - phi_t[:, :dim])
        return jnp.mean((pred_delta - actual_delta)**2)

    def refine(self, state_history: list, g: np.ndarray):
        """Refine structure constants from observed history."""
        if self.params is None:
            return

        history = jnp.array(state_history)
        # Ensure history is real for loss_fn
        history = jnp.real(history)
        g_jnp = jnp.real(jnp.array(g))

        @jax.jit
        def update(params, opt_state, history, g):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, history, g, self.n_gens, self.dim)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        self.params, self.opt_state = update(self.params, self.opt_state, history, g_jnp)
        return [np.array(gen, dtype=complex) for gen in self.params.reshape(self.n_gens, self.dim, self.dim)]
