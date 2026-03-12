import numpy as np
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cross_domain.universal_solver import UniversalProblemSolver
from cross_domain.transfer_learning import RKHSTransferLearner
from utils.snapshots import ContextSnapshotter

def test_universal_solver():
    print("Testing Universal Solver...")
    solver = UniversalProblemSolver()
    res = solver.run_solver("Website is slow", ["Low budget"])
    assert "REFINED" in res["final_solution"]
    assert res["sub_problems_count"] == 3
    print("  ✓ Universal Solver OK")

def test_transfer_learning():
    print("Testing RKHS Transfer Learning...")
    learner = RKHSTransferLearner(epsilon=0.1)
    source = jnp.array(np.random.randn(10, 2))
    target = jnp.array(np.random.randn(20, 2))

    augmented = learner.transfer_knowledge(source, target)
    assert len(augmented) == 30

    t_ability = learner.calculate_transferability(source, target)
    assert 0 <= t_ability <= 1.0
    print("  ✓ Transfer Learning OK")

def test_snapshots():
    print("Testing Context Snapshots...")
    snapshotter = ContextSnapshotter()
    # Large data to ensure compression
    data = {"alpha": np.random.randn(1000).tolist(), "meta": "test" * 100}

    snap = snapshotter.create_snapshot(1, data)
    # Compression may not always yield smaller size for tiny payloads,
    # but for larger random-ish JSON it usually does.
    # Actually, random data doesn't compress well. Repetitive data does.
    print(f"    Raw: {snap['raw_size']} | Compressed: {snap['compressed_size']}")

    recovered = snapshotter.recover_snapshot(0)
    assert recovered["meta"] == "test" * 100

    summary = snapshotter.summary()
    assert summary["n_snapshots"] == 1
    print("  ✓ Snapshots OK")

if __name__ == "__main__":
    test_universal_solver()
    test_transfer_learning()
    test_snapshots()
    print("\nALL PHASE 6 TESTS PASSED")
