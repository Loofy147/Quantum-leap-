import time
import numpy as np
from algebra.lie_expansion import EntanglementBattery, LieAlgebraConfig

def benchmark():
    print("Benchmarking Lie Algebra and Entanglement Battery...")

    start_init = time.time()
    config = LieAlgebraConfig(algebra_dim=4, expansion_order=6)
    battery = EntanglementBattery(config, algebra_type='su_n')
    end_init = time.time()
    print(f"Initialization (including structure constants) took: {end_init - start_init:.4f}s")

    start_evolve = time.time()
    results = battery.evolve(n_steps=50)
    end_evolve = time.time()
    print(f"Evolve (50 steps) took: {end_evolve - start_evolve:.4f}s")

    start_fps = time.time()
    for _ in range(100):
        battery.formal_power_series(epsilon=0.001)
    end_fps = time.time()
    print(f"100 calls to formal_power_series took: {end_fps - start_fps:.4f}s")

if __name__ == "__main__":
    benchmark()
