import unittest
import numpy as np
from src.ekrls import EKRLS
from src.ribbon_filter import RibbonFilter
from src.lie_battery import EntanglementBattery
from src.suffix_smoothing import SuffixSmoothing
from src.metacognition import MetacognitiveMonitor

class TestQuantumSpacetime(unittest.TestCase):
    def test_ekrls_learning(self):
        engine = EKRLS(nu=0.01)
        # Train on simple linear trend
        for i in range(10):
            engine.update(np.array([[i]]), i * 2)

        pred = engine.predict(np.array([[5]]))
        self.assertAlmostEqual(pred, 10.0, delta=1.5)

    def test_ribbon_filter_query(self):
        rf = RibbonFilter(num_keys=100)
        keys = ["k1", "k2", "k3"]
        rf.construct(keys)
        # Note: In our simplified simulation, query always uses the parity match logic.
        # Since construction XORs the fingerprint, it should match.
        self.assertTrue(rf.query("k1"))
        self.assertTrue(rf.query("k2"))

    def test_battery_discharge(self):
        bat = EntanglementBattery(capacity=100)
        amt = bat.discharge(10, np.array([1, 1, 1]))
        self.assertGreater(amt, 0)
        self.assertLess(bat.current_charge, 100)

    def test_suffix_smoothing(self):
        ss = SuffixSmoothing(lmbda=0.5)
        ss.train([("ABC", "T1"), ("BC", "T1")])
        p1 = ss.get_smoothed_probability("ABC", "T1")
        p2 = ss.get_smoothed_probability("BC", "T1")
        self.assertGreater(p1, 0)
        self.assertGreater(p2, 0)

    def test_monitor_bayesian(self):
        mon = MetacognitiveMonitor()
        for i in range(10):
            mon.log_event({'G': 0.9, 'C': 0.9, 'S': 0.9, 'A': 0.9, 'H': 0.9, 'V': 0.9})
        conf = mon.evaluate_q_score_confidence()
        self.assertIsNotNone(conf)
        self.assertAlmostEqual(conf['mean'], 0.9, delta=0.01)

if __name__ == '__main__':
    unittest.main()
