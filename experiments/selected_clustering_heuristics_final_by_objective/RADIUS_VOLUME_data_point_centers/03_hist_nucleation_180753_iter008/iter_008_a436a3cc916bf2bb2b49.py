import numpy as np

class ClusteringHeuristic:
    # Intended mechanism family: Hybrid iterative medoid refinement with radius-based cluster prioritization, bounded medoid replacements, and k-means++ inspired initialization
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers: use k-means++ inspired initialization
        centers = np.zeros((p, X.shape[1]))
        centers[0] = X[rng.choice(X.shape[0], size=1, replace=False)].copy()
        for i in range(1, p):
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=-1)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            idx = rng.choice(X.shape[0], size=1, replace=False, p=probabilities)
            centers[i] = X[idx].copy()

        for _ in range(15):  # Number of iterations
            # Assign each point to its nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=-1)
            nearest_center = np.argmin(distances, axis=1)

            # Compute cluster radii
            radii = np.zeros(p)
            for j in range(p):
                points_in_cluster = X[nearest_center == j]
                if points_in_cluster.size > 0:
                    radii[j] = np.max(np.linalg.norm(points_in_cluster - centers[j], axis=-1))

            # Identify clusters with the largest radius^d contribution
            contributions = radii ** X.shape[1]
            largest_radius_indices = np.argsort(contributions)[::-1]

            # Replace medoids in the worst clusters using bounded medoid replacements
            for j in largest_radius_indices[:p // 2]:  # Replace centers in the worst half of the clusters
                points_in_cluster = X[nearest_center == j]
                if points_in_cluster.size > 1:
                    # Select a new medoid that minimizes the maximum distance to other points in the cluster
                    new_medoid_idx = np.argmin(np.max(np.linalg.norm(points_in_cluster[:, np.newaxis] - points_in_cluster, axis=-1), axis=0))
                    new_medoid = points_in_cluster[new_medoid_idx]
                    # Check if the new medoid reduces the radius
                    new_radius = np.max(np.linalg.norm(points_in_cluster - new_medoid, axis=-1))
                    if new_radius < radii[j]:
                        centers[j] = new_medoid.copy()

        return centers