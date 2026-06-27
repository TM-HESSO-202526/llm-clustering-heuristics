import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Intended mechanism family: Geometric-Medoid Clustering with Voronoi Refinement
        # This method constructs an initial set of medoids using geometric reasoning and then refines them using Voronoi partitioning.

        n, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Initialize medoids geometrically
        medoids = []
        for _ in range(p):
            if not medoids:
                # Choose the point with the maximum average distance to all other points
                dists = np.linalg.norm(X[:, None] - X, axis=2)
                np.fill_diagonal(dists, np.inf)
                avg_dists = np.mean(dists, axis=1)
                medoid_idx = np.argmax(avg_dists)
                medoids.append(X[medoid_idx])
            else:
                # Choose the point with the maximum minimum distance to the existing medoids
                min_dists = np.linalg.norm(X[:, None] - np.array(medoids), axis=2).min(axis=1)
                medoid_idx = np.argmax(min_dists)
                medoids.append(X[medoid_idx])

        # Compute Voronoi partitioning
        dists = np.linalg.norm(X[:, None] - np.array(medoids), axis=2)
        labels = np.argmin(dists, axis=1)

        # Iteratively update medoids based on Voronoi refinement
        for _ in range(10):  # Limited iterations for scalability
            new_medoids = []
            for i in np.unique(labels):
                cluster_points = X[labels == i]
                # Choose the point with the minimum average distance to all other points in the cluster
                avg_dists = np.mean(np.linalg.norm(cluster_points[:, None] - cluster_points, axis=2), axis=1)
                new_medoid_idx = np.argmin(avg_dists)
                new_medoids.append(cluster_points[new_medoid_idx])

            # Check for convergence
            if np.all(np.array(new_medoids) == np.array(medoids)):
                break

            medoids = new_medoids

            # Compute Voronoi partitioning
            dists = np.linalg.norm(X[:, None] - np.array(medoids), axis=2)
            labels = np.argmin(dists, axis=1)

        return np.array(medoids)