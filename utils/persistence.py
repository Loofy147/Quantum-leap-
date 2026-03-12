import numpy as np
import os

class PersistenceManager:
    """
    Tier 2026: Spacetime Manifold Persistence Manager.
    Handles serialization of RKHS structures and Lie Algebra generators.
    """
    @staticmethod
    def save_system_state(filepath: str, states: dict):
        """Saves a dictionary of states to an NPZ file."""
        np.savez(filepath, **states)

    @staticmethod
    def load_system_state(filepath: str) -> dict:
        """Loads states from an NPZ file."""
        if not os.path.exists(filepath):
            return {}
        with np.load(filepath, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
