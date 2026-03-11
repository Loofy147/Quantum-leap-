import numpy as np
import jax.numpy as jnp
import os
import json
import zlib
from typing import Dict, Any

class ContextSnapshotter:
    """
    Tier 2027: Recursive Context Snapshots.
    Maintains compressed chronological records of system state for temporal coherence.
    """
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.snapshots = []

    def create_snapshot(self, step: int, components: Dict[str, Any]) -> dict:
        """
        Creates a compressed snapshot of the essential context.
        components: dictionary of state artifacts (EKRLS, Lie, etc.)
        """
        # Serialization helper
        def serialize(obj):
            if isinstance(obj, (np.ndarray, jnp.ndarray)):
                return obj.tolist()
            return obj

        raw_data = json.dumps(components, default=serialize).encode('utf-8')
        compressed = zlib.compress(raw_data, level=self.compression_level)

        snapshot = {
            "step": step,
            "data": compressed,
            "raw_size": len(raw_data),
            "compressed_size": len(compressed)
        }
        self.snapshots.append(snapshot)
        return snapshot

    def recover_snapshot(self, index: int) -> Dict[str, Any]:
        """Recovers and decompresses a snapshot."""
        if index < 0 or index >= len(self.snapshots):
            return {}

        compressed = self.snapshots[index]["data"]
        raw_data = zlib.decompress(compressed).decode('utf-8')
        return json.loads(raw_data)

    def summary(self) -> dict:
        """Report on snapshot compression efficiency."""
        if not self.snapshots:
            return {}

        total_raw = sum(s["raw_size"] for s in self.snapshots)
        total_comp = sum(s["compressed_size"] for s in self.snapshots)

        return {
            "n_snapshots": len(self.snapshots),
            "total_raw_kb": total_raw / 1024.0,
            "total_compressed_kb": total_comp / 1024.0,
            "compression_ratio": float(total_raw / (total_comp + 1e-12))
        }
