import numpy as np
from scipy.stats import bayes_mvs

class MetacognitiveMonitor:
    def __init__(self, history_limit=100):
        self.performance_history = []
        self.history_limit = history_limit
        self.bias_detected = False

    def log_event(self, q_score_metrics):
        q = (0.18 * q_score_metrics.get('G', 0) +
             0.22 * q_score_metrics.get('C', 0) +
             0.20 * q_score_metrics.get('S', 0) +
             0.18 * q_score_metrics.get('A', 0) +
             0.12 * q_score_metrics.get('H', 0) +
             0.10 * q_score_metrics.get('V', 0))

        self.performance_history.append(float(q))
        if len(self.performance_history) > self.history_limit:
            self.performance_history.pop(0)

    def detect_anchoring_bias(self, new_value, window=5):
        if len(self.performance_history) < window:
            return False

        recent_avg = np.mean(self.performance_history[-window:])
        std_dev = np.std(self.performance_history[-window:])

        if std_dev < 0.00001:
            self.bias_detected = True
            return True
        return False

    def evaluate_q_score_confidence(self):
        if len(self.performance_history) < 2:
            return None

        # Add tiny jitter to avoid bayes_mvs failure on zero variance
        data = np.array(self.performance_history)
        if np.std(data) < 1e-9:
             data = data + np.random.normal(0, 1e-10, size=data.shape)

        try:
            mean_conf, _, _ = bayes_mvs(data, alpha=0.95)
            return {
                "mean": float(mean_conf.statistic),
                "ci_low": float(mean_conf.minmax[0]),
                "ci_high": float(mean_conf.minmax[1])
            }
        except:
            return {"mean": float(np.mean(data)), "ci_low": 0.0, "ci_high": 1.0}

    def check_for_collapse_risk(self):
        if len(self.performance_history) < 10:
            return False

        recent = self.performance_history[-3:]
        previous = self.performance_history[-10:-3]

        if np.mean(recent) < np.mean(previous) * 0.7:
            return True
        return False
