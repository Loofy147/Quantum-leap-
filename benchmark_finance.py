import time
import numpy as np
from cross_domain.finance import FinancialQuantumAnalyzer, generate_market_data

def benchmark():
    print("Benchmarking FinancialQuantumAnalyzer.analyze...")
    mdata = generate_market_data(n=1000)
    analyzer = FinancialQuantumAnalyzer()

    start_time = time.time()
    analyzer.analyze(mdata)
    end_time = time.time()

    print(f"Analyze (1000 steps) took: {end_time - start_time:.4f}s")
    print(f"Average time per step: {(end_time - start_time)/1000:.6f}s")

if __name__ == "__main__":
    benchmark()
