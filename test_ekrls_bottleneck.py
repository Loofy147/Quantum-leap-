import time
import numpy as np
from engines.ekrls_engine import SquareRootEKRLS, EKRLSConfig

def benchmark_ekrls():
    config = EKRLSConfig(window_size=100)
    engine = SquareRootEKRLS(config)

    dim = config.state_dim
    n_steps = 200

    start_time = time.time()
    for i in range(n_steps):
        phi = np.random.randn(dim)
        phi /= np.linalg.norm(phi)
        y = float(np.random.randn())
        engine.update(phi, y)
    end_time = time.time()

    print(f"EKRLS {n_steps} steps took: {end_time - start_time:.4f}s")
    print(f"Average time per step: {(end_time - start_time)/n_steps:.4f}s")

if __name__ == "__main__":
    benchmark_ekrls()
