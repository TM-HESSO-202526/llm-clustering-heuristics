import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Intended mechanism family: Distance-based medoid selection with iterative refinement
        if rng is None:
            rng = np.random.default_rng()

        n, d = X.shape
        centers = np.zeros((p, d))

        # Random initial center
        initial_center_idx = rng.integers(n)
        centers[0] = X[initial_center_idx]

        # Initialize a set to keep track of points that have been selected as centers
        selected_points = set([initial_center_idx])

        for i in range(1, p):
            # Find the point that is farthest from all existing centers
            max_distance = 0
            max_distance_idx = None
            for j in range(n):
                if j not in selected_points:
                    dists = np.linalg.norm(X[j] - centers[:i], axis=1)
                    min_dist = np.min(dists)
                    if min_dist > max_distance:
                        max_distance = min_dist
                        max_distance_idx = j

            # Add this point as the next center
            centers[i] = X[max_distance_idx]
            selected_points.add(max_distance_idx)

        # Distance-based refinement
        for _ in range(5):  # Limited number of refinement iterations
            # Compute distances to all existing centers
            dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

            # For each point, find its nearest center
            nearest_center_idx = np.argmin(dists, axis=1)

            # Compute cluster radii for each center
            cluster_radii = np.zeros(p)
            for i in range(p):
                cluster_points = X[nearest_center_idx == i]
                if len(cluster_points) > 0:
                    cluster_radii[i] = np.max(np.linalg.norm(cluster_points - centers[i], axis=1))

            # Prioritize refinement of centers with the largest cluster radii
            refinement_order = np.argsort(-cluster_radii)

            for i in refinement_order:
                # Find points assigned to the current center
                cluster_points = X[nearest_center_idx == i]

                # If the cluster is not empty, try to refine the center
                if len(cluster_points) > 0:
                    # Compute distances to the current center
                    dists = np.linalg.norm(cluster_points - centers[i], axis=1)

                    # Find points in the cluster with the smallest maximum distance to other points
                    min_max_distance_idx = np.argmin(np.max(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1))
                    refined_center = cluster_points[min_max_distance_idx]

                    # Compute distances to the refined center
                    refined_dists = np.linalg.norm(X - refined_center, axis=1)

                    # Check if the refined center improves the cluster radius
                    refined_cluster_radius = np.max(refined_dists[nearest_center_idx == i])
                    if refined_cluster_radius < cluster_radii[i]:
                        centers[i] = refined_center

        return centers