"""
Metacognitive Layer — الوعي الميتا-معرفي
طبقة مراقبة ذكية للكشف عن انهيار التشابك

Monitors quantum simulation for:
- Anchoring bias in state estimation
- Premature entanglement collapse
- Q-score validation with Bayesian calibration
- Real-time reasoning chain supervision
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import warnings


@dataclass
class MetacognitiveConfig:
    """Configuration for the metacognitive monitoring layer."""
    # Collapse detection
    coherence_collapse_threshold: float = 0.15   # Below → collapse alert
    entropy_spike_threshold: float = 0.5         # Sudden entropy jump → alert
    anchoring_window: int = 20                   # Steps to check for anchoring

    # Q-score thresholds
    q_score_minimum: float = 0.85                # Below → reject model
    q_bayesian_prior_strength: float = 10.0      # Concentration of Dirichlet prior

    # Reasoning chain
    max_circular_depth: int = 3                  # Max identical states before circular flag
    confidence_overcheck_threshold: float = 0.98 # Overconfidence flag

    # Q-score dimension weights (Bayesian calibrated)
    q_weights: dict = field(default_factory=lambda: {
        'G': 0.25,  # Grounding
        'C': 0.20,  # Certainty
        'S': 0.15,  # Structure
        'A': 0.15,  # Applicability
        'Co': 0.15, # Coherence
        'Ge': 0.10, # Generativity
    })


class SpectralBiasDetector:
    """
    Monitors the spectral properties of the EKRLS Gram matrix.
    Detects low-rank collapse and numerical instability.
    """
    def __init__(self, condition_threshold: float = 1e6):
        self.condition_threshold = condition_threshold
        self.spectral_history = []
        self.entropy_history = []
        self.prev_probs: Optional[np.ndarray] = None

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Kullback-Leibler divergence between two distributions."""
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def observe_spectrum(self, eigenvalues: np.ndarray) -> dict:
        """Analyze eigenvalues for rank collapse or ill-conditioning."""
        if len(eigenvalues) < 2:
            return {"status": "INSUFFICIENT_DATA"}

        # Sort desc
        ev = np.sort(np.abs(eigenvalues))[::-1]
        cond = ev[0] / (ev[-1] + 1e-12)

        # Effective rank (number of eigenvalues > 1e-6 * max)
        eff_rank = np.sum(ev > (ev[0] * 1e-6))
        rank_ratio = eff_rank / len(ev)

        status = "HEALTHY"
        if cond > self.condition_threshold:
            status = "ILL_CONDITIONED"
        elif rank_ratio < 0.3:
            status = "RANK_COLLAPSE"

        # Compute surprise via KL-Divergence of normalized spectrum
        probs = ev / (ev.sum() + 1e-12)

        # Phase 5: Spectral Entropy tracking
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        self.entropy_history.append(float(entropy))

        surprise = 0.0
        if self.prev_probs is not None and len(self.prev_probs) == len(probs):
            surprise = self._compute_kl_divergence(probs, self.prev_probs)
        self.prev_probs = probs

        result = {
            "condition_number": float(cond),
            "effective_rank": int(eff_rank),
            "rank_ratio": float(rank_ratio),
            "surprise_signal": surprise,
            "status": status,
            "spectral_entropy": float(entropy)
        }
        self.spectral_history.append(result)
        return result

    def calculate_curvature(self) -> float:
        """
        Phase 5: RKHS Curvature calculation.
        Measures acceleration of spectral entropy shift.
        """
        if len(self.entropy_history) < 3:
            return 0.0
        h1, h2, h3 = self.entropy_history[-3:]
        # Second difference approx
        curvature = abs(h3 - 2*h2 + h1)
        return float(curvature)

class AutonomousRegulator:
    """
    Tier 2026: Autonomous Meta-Regulation.
    Dynamically adjusts EKRLS parameters based on RKHS curvature and Q-Score.
    """
    def __init__(self, curvature_threshold: float = 0.05):
        self.curvature_threshold = curvature_threshold
        self.adjustments_made = 0

    def adjust_configs(self, curvature: float, config, q_score: float) -> dict:
        """Adjusts EKRLS parameters to maintain stability."""
        prev_sigma = config.kernel_sigma
        prev_lam = config.forgetting_factor

        if curvature > self.curvature_threshold:
            # Manifold is shifting rapidly -> Increase bandwidth and adaptation
            config.kernel_sigma = float(np.clip(config.kernel_sigma * 1.1, 0.5, 5.0))
            config.forgetting_factor = float(np.clip(config.forgetting_factor * 0.99, 0.8, 0.999))
            self.adjustments_made += 1
        elif q_score < 0.8:
            # Low grounding -> Increase bandwidth to capture more features
            config.kernel_sigma = float(np.clip(config.kernel_sigma * 1.05, 0.5, 5.0))

        return {
            "curvature": curvature,
            "prev_sigma": prev_sigma,
            "new_sigma": config.kernel_sigma,
            "prev_lam": prev_lam,
            "new_lam": config.forgetting_factor,
            "adjusted": config.kernel_sigma != prev_sigma or config.forgetting_factor != prev_lam
        }

class BiasDetector:
    """
    Detects cognitive/computational biases in quantum simulation.

    Monitored biases:
    - Anchoring: state estimates stuck near initial value
    - Confirmation: only accepting states that match prior hypothesis
    - Circular reasoning: prediction chain loops back to premise
    """

    def __init__(self, config: MetacognitiveConfig):
        self.cfg = config
        self.state_buffer: deque = deque(maxlen=config.anchoring_window)
        self.prediction_buffer: deque = deque(maxlen=50)
        self.anchor_value: Optional[float] = None
        self.detected_biases: list[dict] = []

    def observe(self, state_value: float, prediction: float) -> dict:
        """Record new observation and check for biases."""
        self.state_buffer.append(state_value)
        self.prediction_buffer.append(prediction)

        biases_found = []

        # 1. Anchoring bias: variance of states too low over window
        if len(self.state_buffer) >= self.cfg.anchoring_window:
            variance = float(np.var(list(self.state_buffer)))
            if variance < 1e-5:
                biases_found.append({
                    "type": "anchoring",
                    "severity": "HIGH",
                    "variance": variance,
                    "mitigation": "Apply multi-perspective evaluation: reset prior",
                })

        # 2. Overconfidence: predictions too close to same value
        if len(self.prediction_buffer) >= 10:
            pred_var = float(np.var(list(self.prediction_buffer)))
            mean_pred = float(np.mean(list(self.prediction_buffer)))
            if pred_var < 1e-6 and abs(mean_pred) > 0.01:
                biases_found.append({
                    "type": "overconfidence",
                    "severity": "MEDIUM",
                    "pred_variance": pred_var,
                    "mitigation": "Inject exploration noise; review evidence quantity",
                })

        # 3. Circular reasoning: predictions oscillate with period ≤ 3
        if len(self.prediction_buffer) >= 6:
            recent = list(self.prediction_buffer)[-6:]
            diffs = np.diff(recent)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes >= 4:  # Rapid oscillation
                biases_found.append({
                    "type": "circular_reasoning",
                    "severity": "HIGH",
                    "sign_changes": int(sign_changes),
                    "mitigation": "Switch from forward-chaining to backward-chaining strategy",
                })

        self.detected_biases.extend(biases_found)
        return {
            "biases_detected": biases_found,
            "total_bias_events": len(self.detected_biases),
        }

    def is_stuck(self) -> bool:
        """Return True if reasoning is stuck (circular or anchored)."""
        recent = [b for b in self.detected_biases[-5:] if b["severity"] == "HIGH"]
        return len(recent) >= 2


class QScoreValidator:
    """
    Q-Score Validator with Bayesian Calibration.

    Q = Σᵢ wᵢ · dᵢ

    Bayesian calibration: weights wᵢ are updated via Dirichlet posterior
    as more validation evidence accumulates.

    Threshold: Q ≥ 0.85 → model accepted
    """

    def __init__(self, config: MetacognitiveConfig):
        self.cfg = config
        self.dimensions = list(config.q_weights.keys())
        self.n_dims = len(self.dimensions)

        # Bayesian: Dirichlet prior on weights
        alpha_prior = config.q_bayesian_prior_strength
        self.alpha: np.ndarray = np.array([
            config.q_weights[d] * alpha_prior for d in self.dimensions
        ])

        # History of validated models
        self.validation_history: list[dict] = []

    def _bayesian_weights(self) -> dict[str, float]:
        """Compute posterior weight estimates from Dirichlet."""
        alpha_sum = self.alpha.sum()
        weights = self.alpha / alpha_sum
        return {d: float(w) for d, w in zip(self.dimensions, weights)}

    def validate(self, scores: dict[str, float], model_name: str = "model") -> dict:
        """
        Validate a model given dimension scores.

        scores: dict of {dimension: score} where score ∈ [0,1]
        Returns: validation report with Q-score and acceptance decision.
        """
        weights = self._bayesian_weights()

        # Compute Q-score
        q = sum(weights.get(d, 0.0) * scores.get(d, 0.0) for d in self.dimensions)

        # Uncertainty in Q from Dirichlet posterior variance
        # Var[wᵢ] = αᵢ(α₀ - αᵢ) / (α₀²(α₀+1))
        alpha_0 = self.alpha.sum()
        score_vec = np.array([scores.get(d, 0.0) for d in self.dimensions])
        weight_vec = self.alpha / alpha_0
        q_variance = float(np.sum(
            (self.alpha * (alpha_0 - self.alpha)) / (alpha_0**2 * (alpha_0 + 1))
            * score_vec**2
        ))
        q_std = float(np.sqrt(q_variance))

        accepted = q >= self.cfg.q_score_minimum
        bottleneck = min(scores, key=scores.get) if scores else None

        # Update Bayesian weights if accepted (reinforce high-score dimensions)
        if accepted:
            for i, d in enumerate(self.dimensions):
                self.alpha[i] += scores.get(d, 0.0) * 0.1

        result = {
            "model": model_name,
            "q_score": float(q),
            "q_std": float(q_std),
            "q_confidence_interval": (float(q - 1.96*q_std), float(q + 1.96*q_std)),
            "accepted": accepted,
            "threshold": self.cfg.q_score_minimum,
            "dimension_scores": scores,
            "bayesian_weights": weights,
            "bottleneck_dimension": bottleneck,
            "recommendation": (
                "DEPLOY" if q >= 0.92 else
                "ACCEPT with monitoring" if accepted else
                f"REJECT — improve {bottleneck}"
            ),
        }
        self.validation_history.append(result)
        return result

    def auto_score_quantum_model(
        self,
        ekrls_summary: dict,
        battery_summary: dict,
        qec_summary: dict,
    ) -> dict:
        """Auto-compute Q-scores for the integrated quantum system."""
        rmse = ekrls_summary.get("rmse", 0.5)
        coherence = ekrls_summary.get("mean_coherence", 0.5)
        collapses = ekrls_summary.get("collapse_events", 0)
        total = max(1, ekrls_summary.get("total_steps", 1))
        dict_size = ekrls_summary.get("dictionary_size", 0)

        conservation_ok = battery_summary.get("n_conservation_violations", 0) == 0
        battery_pct = battery_summary.get("capacity_pct", 50) / 100

        qec_total = max(1, qec_summary.get("total_corrections", 1))
        qec_success = qec_summary.get("success_rate", 0.5)
        uncertainty_reduction = qec_summary.get("mean_uncertainty_reduction_pct", 50) / 100

        collapse_rate = collapses / total
        ekrls_active = dict_size > 0

        scores = {
            # G: Grounding — EKRLS accuracy on real quantum mechanics
            'G': float(np.clip(1.0 - rmse * 0.8, 0.4, 1.0)),
            'C': float(np.clip(coherence * 0.8 + (1 - collapse_rate * 5) * 0.2, 0.4, 1.0)),
            'S': float(0.99 if conservation_ok else 0.6),
            'A': float(1.0 if dict_size > 5 else 0.8),
            'Co': float(np.clip(coherence * 0.9 + battery_pct * 0.1, 0.4, 1.0)),
            'Ge': float(np.clip(uncertainty_reduction * 1.5 + 0.3, 0.4, 1.0)),
        }

        return self.validate(scores, model_name="QuantumSpacetimeSystem_v1")


class MetacognitiveLayer:
    """
    Master metacognitive monitoring system.

    Integrates:
    - Bias detection (anchoring, circular, overconfidence)
    - Collapse detection (coherence threshold, entropy spikes)
    - Q-score validation (Bayesian calibrated)
    - Real-time reasoning supervision
    """

    def __init__(self, config: Optional[MetacognitiveConfig] = None):
        self.cfg = config or MetacognitiveConfig()
        self.bias_detector = BiasDetector(self.cfg)
        self.spectral_detector = SpectralBiasDetector()
        self.q_validator = QScoreValidator(self.cfg)
        self.regulator = AutonomousRegulator()

        # Collapse tracking
        self.collapse_alerts: list[dict] = []
        self.prev_entropy: float = 0.0
        self.step_count: int = 0

        # Reasoning chain
        self.reasoning_chain: list[str] = []
        self.intervention_log: list[dict] = []

    def monitor_step(self, engine_result: dict) -> dict:
        """
        Monitor one simulation step from the EKRLS engine.
        Returns metacognitive assessment and any interventions.
        """
        self.step_count += 1
        interventions = []
        alerts = []

        coherence = engine_result.get("coherence", 1.0)
        entropy = engine_result.get("entropy", 0.0)
        battery = engine_result.get("battery_level", 1.0)
        pred_error = engine_result.get("pred_error", 0.0)
        y_pred = engine_result.get("y_pred", 0.0)

        # --- Collapse detection ---
        if coherence < self.cfg.coherence_collapse_threshold:
            alert = {
                "type": "ENTANGLEMENT_COLLAPSE",
                "step": self.step_count,
                "coherence": coherence,
                "severity": "CRITICAL",
                "action": "Halt simulation; reinitialize from last stable checkpoint",
            }
            self.collapse_alerts.append(alert)
            alerts.append(alert)

        # --- Entropy spike detection ---
        delta_entropy = abs(entropy - self.prev_entropy)
        if delta_entropy > self.cfg.entropy_spike_threshold:
            alert = {
                "type": "ENTROPY_SPIKE",
                "step": self.step_count,
                "delta_entropy": delta_entropy,
                "severity": "HIGH",
                "action": "Apply Suffix Smoothing correction; check EKRLS kernel",
            }
            alerts.append(alert)
        self.prev_entropy = entropy

        # --- Spectral analysis & Surprise Monitoring ---
        eigenvalues = engine_result.get("eigenvalues")
        if eigenvalues is not None:
            spec_result = self.spectral_detector.observe_spectrum(eigenvalues)

            # Surprise detection
            if spec_result["surprise_signal"] > 2.0:
                alerts.append({
                    "type": "RECOGNITION_SURPRISE",
                    "surprise_value": spec_result["surprise_signal"],
                    "severity": "HIGH",
                    "action": "Regime shift detected; apply predictive backoff",
                })

            if spec_result["status"] != "HEALTHY":
                alerts.append({
                    "type": "SPECTRAL_INSTABILITY",
                    "status": spec_result["status"],
                    "condition": spec_result["condition_number"],
                    "severity": "MEDIUM",
                    "action": "Increase RBF sigma or regularization"
                })

        # --- Bias detection ---
        bias_result = self.bias_detector.observe(coherence, y_pred if y_pred else 0.0)

        if bias_result["biases_detected"]:
            for bias in bias_result["biases_detected"]:
                interventions.append({
                    "type": "BIAS_INTERVENTION",
                    "bias": bias["type"],
                    "action": bias["mitigation"],
                })

        # --- Anchoring check: is battery stuck? ---
        if battery < 0.05:
            interventions.append({
                "type": "BATTERY_CRITICAL",
                "level": battery,
                "action": "Emergency recharge from entanglement reservoir",
            })

        # --- Reasoning stuck check ---
        if self.bias_detector.is_stuck():
            interventions.append({
                "type": "REASONING_STUCK",
                "action": "Switch strategy: backward-chaining from target state",
            })

        self.intervention_log.extend(interventions)

        return {
            "step": self.step_count,
            "alerts": alerts,
            "interventions": interventions,
            "bias_events": bias_result["total_bias_events"],
            "collapse_alerts": len(self.collapse_alerts),
            "system_ok": len(alerts) == 0 and len(interventions) == 0,
        }

    def run_full_validation(
        self,
        ekrls_summary: dict,
        battery_summary: dict,
        qec_summary: dict,
    ) -> dict:
        """Run complete system validation and produce final Q-score report."""
        q_result = self.q_validator.auto_score_quantum_model(
            ekrls_summary, battery_summary, qec_summary
        )

        return {
            "q_validation": q_result,
            "total_monitoring_steps": self.step_count,
            "total_collapse_alerts": len(self.collapse_alerts),
            "total_bias_events": len(self.bias_detector.detected_biases),
            "total_interventions": len(self.intervention_log),
            "system_stability": 1.0 - len(self.collapse_alerts) / max(1, self.step_count),
            "metacognitive_overhead_pct": 100 * len(self.intervention_log) / max(1, self.step_count),
        }

    def confidence_calibration_report(self) -> dict:
        """
        Report on confidence calibration quality.
        Flags if system is overconfident or underconfident.
        """
        if not self.q_validator.validation_history:
            return {"status": "No validations performed yet"}

        q_scores = [v["q_score"] for v in self.q_validator.validation_history]
        accepted = [v["accepted"] for v in self.q_validator.validation_history]

        return {
            "n_validations": len(q_scores),
            "mean_q_score": float(np.mean(q_scores)),
            "acceptance_rate": float(np.mean(accepted)),
            "q_score_std": float(np.std(q_scores)),
            "bayesian_weight_evolution": self.q_validator._bayesian_weights(),
            "overconfidence_events": sum(
                1 for b in self.bias_detector.detected_biases
                if b["type"] == "overconfidence"
            ),
        }
