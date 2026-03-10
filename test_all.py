#!/usr/bin/env python3
"""
Test Suite — مجموعة الاختبارات الشاملة
للنظام الهندسي للزمكان الناشئ

Tests: EKRLS | Ribbon | Lie Algebra | Suffix Smoothing | Metacognitive | Integration
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from engines.ekrls_engine import EKRLSQuantumEngine, EKRLSConfig, SquareRootEKRLS, RBFKernel
from filters.ribbon_filter import RibbonFilter, RibbonConfig, EntanglementIndex, generate_entanglement_pairs
from algebra.lie_expansion import EntanglementBattery, LieAlgebraConfig, LieAlgebra
from error_correction.suffix_smoothing import QuantumSuffixSmoother, SuffixConfig, QuantumErrorCorrector
from metacognition.metacognitive_layer import MetacognitiveLayer, MetacognitiveConfig, QScoreValidator, SpectralBiasDetector

PASSED = 0
FAILED = 0
RESULTS = []

def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  ✅ {name}")
        PASSED += 1
        RESULTS.append({"test": name, "status": "PASS"})
    except AssertionError as e:
        print(f"  ❌ {name}: {e}")
        FAILED += 1
        RESULTS.append({"test": name, "status": "FAIL", "error": str(e)})
    except Exception as e:
        print(f"  💥 {name}: {type(e).__name__}: {e}")
        FAILED += 1
        RESULTS.append({"test": name, "status": "ERROR", "error": str(e)})


# ═══════════════════════════════════════════════════════
# SECTION 1: RBF Kernel
# ═══════════════════════════════════════════════════════
print("\n[Section 1] RBF Kernel Tests")

def test_kernel_self_similarity():
    k = RBFKernel(sigma=1.0)
    x = np.array([1.0, 0.0])
    assert abs(k(x, x) - 1.0) < 1e-10

def test_kernel_symmetry():
    k = RBFKernel(sigma=1.0)
    x, y = np.random.randn(2), np.random.randn(2)
    assert abs(k(x, y) - k(y, x)) < 1e-10

def test_kernel_positive_definite():
    k = RBFKernel(sigma=0.5)
    X = np.random.randn(10, 2)
    K = k.gram_matrix(X)
    ev = np.linalg.eigvalsh(K)
    assert np.all(ev > -1e-12)

def test_kernel_decay():
    k = RBFKernel(sigma=1.0)
    x = np.zeros(2)
    y1 = np.array([1.0, 0.0])
    y2 = np.array([2.0, 0.0])
    assert k(x, y1) > k(x, y2)

test("kernel_self_similarity", test_kernel_self_similarity)
test("kernel_symmetry", test_kernel_symmetry)
test("kernel_positive_definite", test_kernel_positive_definite)
test("kernel_decay_with_distance", test_kernel_decay)


# ═══════════════════════════════════════════════════════
# SECTION 2: EKRLS Engine
# ═══════════════════════════════════════════════════════
print("\n[Section 2] EKRLS Engine Tests")

def test_ekrls_steps():
    engine = EKRLSQuantumEngine(EKRLSConfig(state_dim=2))
    for i in range(30):
        phi = np.random.randn(2)
        engine.step(phi, 1.0)
    assert len(engine.state_history) == 30

def test_ekrls_rmse():
    engine = EKRLSQuantumEngine(EKRLSConfig(state_dim=2))
    # Target: y = constant 0.5
    for _ in range(50):
        phi = np.random.randn(2)
        engine.step(phi, 0.5)
    s = engine.summary()
    assert s["rmse"] < 0.2

def test_ekrls_tracking():
    engine = EKRLSQuantumEngine(EKRLSConfig(state_dim=2))
    # High coherence state
    phi = np.array([1.0, 1.0]) / np.sqrt(2)
    res = engine.step(phi, 1.0)
    assert res["coherence"] > 0.4

test("ekrls_updates_30_steps", test_ekrls_steps)
test("ekrls_rmse_reasonable", test_ekrls_rmse)
test("ekrls_state_tracking", test_ekrls_tracking)


# ═══════════════════════════════════════════════════════
# SECTION 3: Ribbon Filter
# ═══════════════════════════════════════════════════════
print("\n[Section 3] Ribbon Filter Tests")

def test_ribbon_build_query():
    cfg = RibbonConfig(n_keys=100, fp_rate=0.01)
    rf = RibbonFilter(cfg)
    keys = [os.urandom(8) for _ in range(50)]
    rf.build(keys)
    for k in keys:
        assert rf.query(k)

def test_entanglement_index():
    idx = EntanglementIndex(expected_pairs=100)
    pairs = [(1, 2, 0), (3, 4, 10)]
    idx.build_from_pairs(pairs)
    assert idx.is_entangled(1, 2, 0)
    assert idx.is_entangled(3, 4, 10)

test("ribbon_no_false_negatives", test_ribbon_build_query)
test("entanglement_index_build", test_entanglement_index)


# ═══════════════════════════════════════════════════════
# SECTION 4: Lie Algebra & Battery
# ═══════════════════════════════════════════════════════
print("\n[Section 4] Lie Algebra & Battery Tests")

def test_su_n_generators():
    """su(n) generators must be anti-Hermitian and traceless."""
    for n in [2, 3]:
        algebra = LieAlgebra('su_n', n=n)
        assert len(algebra.generators) == n**2 - 1
        for i, X in enumerate(algebra.generators):
            err = np.max(np.abs(X + X.conj().T))
            assert err < 1e-10, f"Generator {i} not anti-Hermitian"
            assert abs(np.trace(X)) < 1e-10, f"Generator {i} not traceless"

def test_battery_rk4():
    battery = EntanglementBattery(LieAlgebraConfig(battery_capacity=10.0))
    # Ensure it runs without error (integrated RK4)
    battery.evolve(n_steps=10)
    assert battery.E_battery > 0

test("su_n_generators_valid", test_su_n_generators)
test("battery_rk4_evolution", test_battery_rk4)


# ═══════════════════════════════════════════════════════
# SECTION 5: Metacognitive Layer
# ═══════════════════════════════════════════════════════
print("\n[Section 5] Metacognitive Layer Tests")

def test_spectral_bias_detection():
    """Spectral detector must identify ill-conditioned matrices."""
    detector = SpectralBiasDetector(condition_threshold=100)
    # Ill-conditioned eigenvalues
    ev_bad = np.array([1000.0, 0.001])
    res = detector.observe_spectrum(ev_bad)
    assert res["status"] == "ILL_CONDITIONED"

    # Rank collapse (ratio < 0.3)
    detector_rank = SpectralBiasDetector(condition_threshold=1e20)
    ev_collapse = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    res = detector_rank.observe_spectrum(ev_collapse)
    assert res["status"] == "RANK_COLLAPSE"

def test_q_score_threshold():
    validator = QScoreValidator(MetacognitiveConfig())
    scores = {'G': 0.9, 'C': 0.9, 'S': 0.9, 'A': 0.9, 'Co': 0.9, 'Ge': 0.9}
    result = validator.validate(scores)
    assert result["accepted"]

test("spectral_bias_detection", test_spectral_bias_detection)
test("q_score_accepts_high_scores", test_q_score_threshold)


# ═══════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  FINAL RESULTS: {PASSED} passed, {FAILED} failed")
print(f"  Pass rate: {100*PASSED/(PASSED+FAILED):.1f}%")
print("=" * 60)

if FAILED > 0:
    sys.exit(1)
