import numpy as np
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engines.ekrls_engine import SquareRootEKRLS, EKRLSConfig
from algebra.lie_expansion import LieAlgebra
from metacognition.metacognitive_layer import SpectralBiasDetector, AutonomousRegulator
from algebra.differentiable_physics import compute_vjp_update
from utils.persistence import PersistenceManager

def test_persistence():
    print("Testing persistence...")
    cfg = EKRLSConfig(state_dim=2)
    ekrls = SquareRootEKRLS(cfg)

    # Fake some state
    ekrls._dict_X = [np.array([1.0, 0.0])]
    ekrls._dict_y = [0.5]
    ekrls._R_sqrt = np.eye(1)
    ekrls._alpha = np.array([0.5])
    ekrls.update_count = 1

    state = ekrls.save_state()
    path = "test_ekrls.npz"
    PersistenceManager.save_system_state(path, {"ekrls": state})

    loaded = PersistenceManager.load_system_state(path)
    new_ekrls = SquareRootEKRLS(cfg)
    new_ekrls.load_state(loaded["ekrls"].item())

    assert np.array_equal(new_ekrls._dict_X, ekrls._dict_X)
    assert new_ekrls.update_count == 1
    os.remove(path)
    print("  ✓ Persistence OK")

def test_regulation():
    print("Testing meta-regulation...")
    detector = SpectralBiasDetector()
    # Mock curvature spike
    detector.entropy_history = [1.0, 1.1, 1.3]
    curvature = detector.calculate_curvature()
    assert curvature > 0

    regulator = AutonomousRegulator(curvature_threshold=0.01)
    cfg = EKRLSConfig(kernel_sigma=1.0)
    res = regulator.adjust_configs(curvature, cfg, 0.9)

    assert cfg.kernel_sigma > 1.0
    assert res["adjusted"] is True
    print("  ✓ Regulation OK")

def test_vjp():
    print("Testing VJP update...")
    phi_t = jnp.array([1.0, 0.0])
    phi_next = jnp.array([1.1, 0.1])
    g = jnp.array([1.0, 0.0, 0.0])
    # su(2) gens mock
    gens = [jnp.array([[0, 1], [-1, 0]]), jnp.array([[0, 1j], [1j, 0]]), jnp.array([[1j, 0], [0, -1j]])]

    grads = compute_vjp_update(gens, g, phi_t, phi_next)
    assert grads.shape == (3, 2, 2)
    print("  ✓ VJP OK")

if __name__ == "__main__":
    test_persistence()
    test_regulation()
    test_vjp()
    print("\nALL PHASE 5 TESTS PASSED")
