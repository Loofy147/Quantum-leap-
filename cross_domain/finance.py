"""
Finance Domain Adapter
========================
تطبيق المحركات الكمومية على أسواق المال

Quantum → Finance Isomorphism:
  Φ_n  (quantum state)    →  [price, volume, bid-ask, momentum]  (market state)
  coherence               →  market stability (low vol)
  entanglement collapse   →  volatility spike / regime break
  EKRLS in RKHS           →  non-linear volatility surface estimation
  Viterbi over QEC codes  →  hidden market regime sequence (bull/bear/range/crash)
  Entanglement battery     →  liquidity reserve (absorbs/releases volatility)
  Ribbon filter           →  fast lookup of historical pattern signatures

Key advantage over classical models:
  - GARCH is LINEAR in variance → misses regime-dependent non-linearity
  - EKRLS maps into RKHS → captures volatility clustering, skew, fat tails
  - Battery model: liquidity acts as buffer; its depletion = flash crash precursor
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engines.ekrls_engine import EKRLSQuantumEngine, EKRLSConfig
from error_correction.suffix_smoothing import QuantumErrorCorrector, SuffixConfig
from algebra.lie_expansion import EntanglementBattery, LieAlgebraConfig
from metacognition.metacognitive_layer import MetacognitiveLayer, MetacognitiveConfig


REGIME_LABELS = {
    0: "BULL_QUIET",   1: "BULL_VOLATILE",
    2: "RANGE_LOW",    3: "RANGE_HIGH",
    4: "BEAR_SLOW",    5: "BEAR_PANIC",
    6: "CRASH",        7: "RECOVERY",
}


def generate_market_data(n: int = 200, seed: int = 42) -> dict:
    """
    Generate synthetic OHLCV market data with regime changes.
    Returns dict of price, volume, returns, realized_vol arrays.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(n)
    prices[0] = 100.0
    regimes = np.zeros(n, dtype=int)

    # Regime schedule: bull→range→bear→recovery
    for i in range(1, n):
        if i < 50:   # Bull
            mu, sigma, regime = 0.001, 0.012, 0
        elif i < 90:  # Range
            mu, sigma, regime = 0.0, 0.008, 2
        elif i < 130: # Bear
            mu, sigma, regime = -0.002, 0.018, 4
        elif i < 150: # Crash
            mu, sigma, regime = -0.015, 0.045, 6
        else:         # Recovery
            mu, sigma, regime = 0.003, 0.020, 7

        ret = rng.normal(mu, sigma)
        prices[i] = prices[i-1] * (1 + ret)
        regimes[i] = regime

    returns = np.diff(np.log(prices))
    vol = np.array([
        float(np.std(returns[max(0,i-20):i+1])) if i > 0 else 0.01
        for i in range(len(returns))
    ])
    volume = rng.lognormal(mean=14.0, sigma=0.3, size=n)

    return {
        "prices": prices,
        "returns": returns,
        "realized_vol": vol,
        "volume": volume,
        "true_regimes": regimes,
        "n": n,
    }


def encode_market_state(prices: np.ndarray, returns: np.ndarray,
                         vol: np.ndarray, volume: np.ndarray,
                         i: int, lookback: int = 14) -> np.ndarray:
    """
    Encode market observation into 6D quantum-style state vector Φ.

    Φ = [momentum, vol_z, vol_z2, trend, RSI, volatility_clustering]
    """
    if i < lookback:
        return np.zeros(6)

    # Price momentum (normalized)
    momentum = float(np.sum(returns[max(0,i-5):i])) / (5 * 0.01 + 1e-8)
    momentum = np.clip(momentum, -5, 5) / 5.0

    # Volatility z-score
    vol_mean = float(np.mean(vol[max(0,i-lookback):i]))
    vol_std = float(np.std(vol[max(0,i-lookback):i])) + 1e-8
    vol_z = np.clip((vol[i-1] - vol_mean) / vol_std, -3, 3) / 3.0

    # Volume z-score
    vol_m = float(np.mean(volume[max(0,i-lookback):i]))
    vol_s = float(np.std(volume[max(0,i-lookback):i])) + 1e-8
    vol_z2 = np.clip((volume[i] - vol_m) / vol_s, -3, 3) / 3.0

    # Trend strength (R² of price over window)
    window_prices = prices[max(0,i-lookback):i]
    if len(window_prices) > 2:
        x = np.arange(len(window_prices), dtype=float)
        x -= x.mean()
        y = window_prices - window_prices.mean()
        r2 = float((np.dot(x,y)**2) / (np.dot(x,x)*np.dot(y,y) + 1e-12))
    else:
        r2 = 0.0

    # RSI (Relative Strength Index) proxy
    diffs = returns[max(0,i-lookback):i]
    ups = diffs[diffs > 0]
    downs = -diffs[diffs < 0]
    avg_up = np.mean(ups) if len(ups) > 0 else 0
    avg_down = np.mean(downs) if len(downs) > 0 else 1e-8
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    rsi_norm = (rsi - 50) / 50.0 # Normalized [-1, 1]

    # Volatility clustering (Autocorrelation of returns magnitude)
    abs_ret = np.abs(returns[max(0,i-lookback):i])
    if len(abs_ret) > 5:
        corr = float(np.corrcoef(abs_ret[1:], abs_ret[:-1])[0,1]) if np.std(abs_ret) > 1e-8 else 0.0
    else:
        corr = 0.0

    return np.array([momentum, vol_z, vol_z2, r2, rsi_norm, corr])



def get_batch_market_features(prices, returns, vol, volume, lookback=14):
    """
    Vectorized extraction of market features for all time steps (Bolt ⚡ Optimized).
    Returns (N, 6) array of features.
    """
    import pandas as pd
    n = len(prices)
    ret_series = pd.Series(returns)
    vol_series = pd.Series(vol)
    volu_series = pd.Series(volume)
    price_series = pd.Series(prices)

    # 1. Momentum (5-day)
    mom = ret_series.rolling(window=5).sum() / (5 * 0.01 + 1e-8)
    momentum = (np.clip(mom, -5, 5) / 5.0).fillna(0).values

    # 2. Volatility z-score
    vol_m = vol_series.rolling(window=lookback).mean()
    vol_s = vol_series.rolling(window=lookback).std() + 1e-8
    vol_z = (np.clip((vol_series - vol_m) / vol_s, -3, 3) / 3.0).fillna(0).values

    # 3. Volume z-score
    volu_m = volu_series.rolling(window=lookback).mean()
    volu_s = volu_series.rolling(window=lookback).std() + 1e-8
    volu_z2 = (np.clip((volu_series - volu_m) / volu_s, -3, 3) / 3.0).fillna(0).values

    # 4. Trend strength (R2) via correlation squared
    r = price_series.rolling(window=lookback).corr(pd.Series(np.arange(n)))
    r2 = (r**2).fillna(0).values

    # 5. RSI proxy
    ups = ret_series.clip(lower=0)
    downs = (-ret_series).clip(lower=0)
    avg_up = ups.rolling(window=lookback).mean()
    avg_down = downs.rolling(window=lookback).mean() + 1e-8
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    rsi_norm = ((rsi - 50) / 50.0).fillna(0).values

    # 6. Volatility clustering
    abs_ret = ret_series.abs()
    corr = abs_ret.rolling(window=lookback).corr(abs_ret.shift(1)).fillna(0).values

    features = np.zeros((n, 6))

    # Aligning with original step logic
    # prices: n, returns: n-1, vol: n-1, volume: n
    # momentum, rsi, corr come from returns (n-1)
    # vol_z comes from vol (n-1)
    # volu_z2 comes from volume (n)
    # r2 comes from prices (n)

    # original encode_market_state(i) used returns[i-lookback:i], vol[i-lookback:i], etc.
    # so momentum[i-1] matches sum(returns[i-5:i]).
    # vol_z[i-1] matches mean(vol[i-W:i]).

    features[1:len(momentum)+1, 0] = momentum
    features[1:len(vol_z)+1, 1] = vol_z
    features[1:len(volu_z2), 2] = volu_z2[1:] # Aligning volume[i]
    features[1:len(r2), 3] = r2[1:]
    features[1:len(rsi_norm)+1, 4] = rsi_norm
    features[1:len(corr)+1, 5] = corr

    return features


class FinancialQuantumAnalyzer:
    """
    Quantum-inspired financial analysis engine.

    Uses EKRLS for non-linear volatility estimation and
    Viterbi + Suffix Smoother for market regime detection.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ekrls = EKRLSQuantumEngine(EKRLSConfig(
            state_dim=6,
            kernel_sigma=0.5,    # Tighter kernel: captures local vol clustering
            forgetting_factor=0.97,
            process_noise=0.005,
            measurement_noise=0.02,
            window_size=30,
        ))
        self.regime_detector = QuantumErrorCorrector(SuffixConfig(
            max_suffix_length=5,
            n_qec_codes=8,       # 8 market regimes
        ))
        self.liquidity_battery = EntanglementBattery(
            LieAlgebraConfig(battery_capacity=1.0, coupling_alpha=0.01),
            algebra_type='galilei',
        )
        self.metacog = MetacognitiveLayer(MetacognitiveConfig(
            coherence_collapse_threshold=0.15,
        ))
        self.results: list[dict] = []
        self._regime_initialized = False

    def _vol_from_coherence(self, coherence: float, base_vol: float) -> float:
        """
        Isomorphism: coherence ↔ market stability
        Low coherence → high volatility
        coherence=1.0 → vol = base_vol
        coherence=0.0 → vol = 5x base_vol
        """
        vol_multiplier = 1.0 + 4.0 * (1.0 - coherence)
        return float(base_vol * vol_multiplier)

    def analyze(self, market_data: dict) -> dict:
        """Run full analysis on market data (Bolt ⚡ Optimized)."""
        prices = market_data["prices"]
        returns = market_data["returns"]
        vol = market_data["realized_vol"]
        volume = market_data["volume"]
        n = market_data["n"]

        # Initialize regime detector
        # Initialize regime detector once (Bolt ⚡ Optimization)
        if not self._regime_initialized:
            self.regime_detector.initialize(n_training=400, seed=self.seed)
            self._regime_initialized = True

        # Batch compute all features (Bolt ⚡ Optimization)
        all_phis = get_batch_market_features(prices, returns, vol, volume)

        vol_forecasts = []
        regime_sequence = []
        liquidity_levels = []
        anomaly_scores = []
        signals = []

        for i in range(1, min(n-1, len(returns)+1)):
            # Use pre-computed Φ
            phi = all_phis[i]
            # Measurement = realized return at this step
            measurement = float(returns[i-1]) if i-1 < len(returns) else 0.0

            # EKRLS step
            step_result = self.ekrls.step(phi, measurement)

            # Volatility forecast from coherence
            coherence = step_result["coherence"]
            base_vol = vol[i-1] if i > 0 and i-1 < len(vol) else 0.01
            vol_forecast = self._vol_from_coherence(coherence, base_vol)
            vol_forecasts.append(vol_forecast)

            # Market regime detection via QEC → Viterbi codes
            qec_result = self.regime_detector.correct(phi)
            regime_code = qec_result["qec_code"] % 8
            regime_label = REGIME_LABELS[regime_code]
            regime_sequence.append(regime_label)

            # Liquidity battery: charge on calm, discharge on stress
            if coherence > 0.7:
                self.liquidity_battery.charge(0.005)
            else:
                stress = 1.0 - coherence
                self.liquidity_battery.discharge(0.01 * stress)
            liquidity_levels.append(self.liquidity_battery.E_battery)

            # Anomaly score: deviation of predicted vol from realized
            pred_error = abs(step_result.get("pred_error", 0.0))
            anomaly_score = float(np.clip(pred_error / (base_vol + 1e-8), 0, 10))
            anomaly_scores.append(anomaly_score)

            # Trading signal from state
            battery = step_result["battery_level"]
            confidence = qec_result["confidence"]
            signal = self._generate_signal(
                momentum=float(phi[0]),
                vol_z=float(phi[1]),
                coherence=coherence,
                regime=regime_code,
                battery=battery,
                confidence=confidence,
            )
            signals.append(signal)

            # Metacognitive monitoring
            self.metacog.monitor_step(step_result)

            self.results.append({
                "i": i,
                "vol_forecast": vol_forecast,
                "regime": regime_label,
                "regime_code": regime_code,
                "liquidity": self.liquidity_battery.E_battery,
                "anomaly_score": anomaly_score,
                "signal": signal,
                "coherence": coherence,
            })

        return {
            "n_analyzed": len(self.results),
            "vol_forecasts": vol_forecasts,
            "regime_sequence": regime_sequence,
            "liquidity_levels": liquidity_levels,
            "anomaly_scores": anomaly_scores,
            "signals": signals,
        }
    def _generate_signal(self, momentum: float, vol_z: float,
                         coherence: float, regime: int,
                         battery: float, confidence: float) -> dict:
        """
        Generate trading signal from quantum-derived features.

        Rules:
        - High coherence + positive momentum + adequate liquidity → LONG
        - Low coherence + negative momentum + crash regime → SHORT/HEDGE
        - Very low coherence → FLAT (uncertainty too high)
        """
        if coherence < 0.2:
            action = "FLAT"
            strength = 0.0
        elif regime == 6:  # CRASH
            action = "SHORT"
            strength = -0.8 * confidence
        elif coherence > 0.6 and momentum > 0.2 and battery > 0.3:
            action = "LONG"
            strength = min(1.0, momentum * coherence * confidence)
        elif coherence > 0.4 and momentum < -0.3:
            action = "SHORT"
            strength = -min(1.0, abs(momentum) * confidence)
        else:
            action = "FLAT"
            strength = 0.0

        return {"action": action, "strength": round(strength, 4), "confidence": round(confidence, 4)}

    def performance_summary(self) -> dict:
        """Compute summary statistics of the analysis."""
        if not self.results:
            return {}

        regimes = [r["regime"] for r in self.results]
        unique_regimes = list(set(regimes))
        regime_counts = {r: regimes.count(r) for r in unique_regimes}

        signals = [r["signal"]["action"] for r in self.results]
        anomalies = [r for r in self.results if r["anomaly_score"] > 3.0]

        coherences = [r["coherence"] for r in self.results]
        vols = [r["vol_forecast"] for r in self.results]

        return {
            "periods_analyzed": len(self.results),
            "regime_distribution": regime_counts,
            "anomaly_events": len(anomalies),
            "mean_vol_forecast": round(float(np.mean(vols)), 5),
            "mean_coherence": round(float(np.mean(coherences)), 3),
            "signal_breakdown": {
                "LONG": signals.count("LONG"),
                "SHORT": signals.count("SHORT"),
                "FLAT": signals.count("FLAT"),
            },
            "ekrls_rmse": round(self.ekrls.summary().get("rmse", 0), 5),
            "battery_final": round(self.liquidity_battery.E_battery, 4),
            "collapse_events": self.ekrls.summary().get("collapse_events", 0),
        }

import pandas as pd

def load_kaggle_market_data(file_path: str, company: str = 'Tesla') -> dict:
    """
    Load and parse market data from Kaggle CSV (Bolt ⚡ Optimized).
    Supports both synthetic format and standard OHLCV Kaggle datasets.
    """
    df = pd.read_csv(file_path)

    # Standardize column names (handle case sensitivity)
    df.columns = [c.strip().capitalize() if c.lower() in ['date', 'open', 'high', 'low', 'close', 'volume'] else c for c in df.columns]

    # Filter by company if column exists
    if 'Company' in df.columns:
        df = df[df['Company'] == company]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    if df.empty:
        return generate_market_data(n=200)

    prices = df['Close'].values
    returns = np.diff(np.log(prices))
    returns_series = pd.Series(returns)

    # Vectorized realized volatility (Bolt ⚡ Optimization)
    vol = returns_series.rolling(window=21, min_periods=1).std().fillna(0.01).values

    volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(prices))

    # Map Trend to numeric regimes
    if 'Trend' in df.columns:
        trend_map = {'Stable': 2, 'Bullish': 0, 'Bearish': 4}
        regimes = np.array([trend_map.get(t, 2) for t in df['Trend'].values])
    else:
        # Vectorized proxy regimes (Bolt ⚡ Optimization)
        # Using 5-day rolling sum to determine regimes
        rolling_ret_5 = returns_series.rolling(window=5).sum()
        regimes = np.full(len(prices), 2, dtype=int)

        # Shift and mask for alignment (original loop started at i=6)
        # We start from the index 5 of returns_series (which corresponds to 6th price)
        mask_bull = (rolling_ret_5 > 0.02).values
        mask_bear = (rolling_ret_5 < -0.02).values

        # Align masks with prices array (prices is 1 element longer than returns)
        regimes[1:][mask_bull] = 0
        regimes[1:][mask_bear] = 4

    return {
        "prices": prices,
        "returns": returns,
        "realized_vol": vol,
        "volume": volume,
        "true_regimes": regimes,
        "n": len(prices),
        "source": f"Kaggle:{company}"
    }
