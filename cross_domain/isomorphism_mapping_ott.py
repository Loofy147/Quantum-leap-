import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

class SpacetimeIsomorphismMapper:
    """
    Phase 3 (Tier 2026): Isomorphism Mapping with Optimal Transport
    Quantifies structural "distance" between different domains.
    """
    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.solver = sinkhorn.Sinkhorn()

    def calculate_distance(self, domain_a_history: jnp.ndarray, domain_b_history: jnp.ndarray) -> float:
        """Calculate the Sinkhorn distance between two domain state distributions."""
        # Setup point cloud geometry
        geom = pointcloud.PointCloud(domain_a_history, domain_b_history, epsilon=self.epsilon)

        # Marginal distributions (uniform)
        a = jnp.ones(len(domain_a_history)) / len(domain_a_history)
        b = jnp.ones(len(domain_b_history)) / len(domain_b_history)

        prob = linear_problem.LinearProblem(geom, a, b)

        # Solve Entropic Regularized Optimal Transport
        out = self.solver(prob)

        return float(out.reg_ot_cost)

    def map_isomorphism(self, source_states: jnp.ndarray, target_states: jnp.ndarray) -> jnp.ndarray:
        """Map the source domain states to the target domain using the entropic plan."""
        geom = pointcloud.PointCloud(source_states, target_states, epsilon=self.epsilon)
        a = jnp.ones(len(source_states)) / len(source_states)
        b = jnp.ones(len(target_states)) / len(target_states)

        prob = linear_problem.LinearProblem(geom, a, b)
        out = self.solver(prob)

        # P = coupling matrix (N x M)
        P = out.matrix

        # Mapped states: Φ_mapped = P @ Φ_target * N (rescale for weights)
        mapped = jnp.dot(P, target_states) * len(source_states)

        return mapped


class SpacetimeManifoldProjector:
    """
    Tier 2026: Spacetime Manifold Projection
    Projects cross-domain states into a shared latent manifold for visualization.
    """
    def __init__(self, target_dim: int = 2):
        self.target_dim = target_dim

    def project_to_manifold(self, domain_states: dict) -> dict:
        """
        Projects multiple domains into a shared manifold using PCA/UMAP logic.
        domain_states: { 'domain_name': jnp.ndarray (T, D) }
        """
        from sklearn.decomposition import PCA
        import numpy as np

        all_states = []
        domain_ranges = {}
        curr = 0

        for name, states in domain_states.items():
            s_np = np.array(states)
            all_states.append(s_np)
            domain_ranges[name] = (curr, curr + len(s_np))
            curr += len(s_np)

        if not all_states: return {}

        combined = np.vstack(all_states)
        pca = PCA(n_components=min(self.target_dim, combined.shape[1], combined.shape[0]))
        projected = pca.fit_transform(combined)

        results = {}
        for name, (start, end) in domain_ranges.items():
            results[name] = projected[start:end].tolist()

        return {
            "projections": results,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "manifold_type": "PCA-RKHS-Shared"
        }
