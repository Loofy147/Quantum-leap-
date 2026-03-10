import time
import numpy as np
from engines.ekrls_engine import SquareRootEKRLS, EKRLSConfig

def benchmark_ekrls():
    # Larger window to make the list-to-array conversion more noticeable
    config = EKRLSConfig(window_size=200, state_dim=10)
    engine = SquareRootEKRLS(config)

    dim = config.state_dim
    n_steps = 1000

    # Warmup
    for _ in range(200):
        phi = np.random.randn(dim)
        phi /= np.linalg.norm(phi)
        y = float(np.random.randn())
        engine.update(phi, y)

    start_time = time.time()
    for i in range(n_steps):
        phi = np.random.randn(dim)
        phi /= np.linalg.norm(phi)
        y = float(np.random.randn())
        engine.update(phi, y)
    end_time = time.time()

    # print(f"EKRLS {n_steps} steps took: {end_time - start_time:.4f}s")
    print(f"Average time per step: {(end_time - start_time)/n_steps:.6f}s")

if __name__ == "__main__":
    benchmark_ekrls()
