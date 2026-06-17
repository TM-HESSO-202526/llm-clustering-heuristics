import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Main mechanism: Hybrid cluster radius reduction with adaptive refinement
        # and bounded medoid replacements to control high-radius clusters.

        # Initialize centers as random points from X, ensuring diversity
        if rng is None:
            rng = np.random.default_rng()
        centers = self._initialize_diverse_centers(X, p, rng)

        # Hierarchical refinement
        for _ in range(15):  # Number of refinement iterations
            # Assign points to nearest centers
            dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(dists, axis=1)

            # Compute cluster radii
            radii = np.zeros(p)
            for i in range(p):
                cluster_points = X[labels == i]
                if cluster_points.size > 0:
                    radii[i] = np.max(np.linalg.norm(cluster_points - centers[i], axis=1))

            # Identify clusters with largest radii
            max_radius_indices = np.argsort(radii)[::-1][:int(p/3)]

            # Refine centers by finding medoids that reduce the largest cluster radii
            for idx in max_radius_indices:
                cluster_points = X[labels == idx]
                if cluster_points.size > 0:
                    new_center_idx = self._find_better_medoid(cluster_points, centers[idx], rng)
                    if new_center_idx is not None:
                        new_center = cluster_points[new_center_idx]
                        centers[idx] = new_center

            # Adaptively split large clusters
            for idx in max_radius_indices:
                cluster_points = X[labels == idx]
                if cluster_points.size > 0 and radii[idx] > np.mean(radii):
                    split_point_idx = np.argmax(np.linalg.norm(cluster_points - centers[idx], axis=1))
                    new_center = cluster_points[split_point_idx]
                    # Replace the center with the new center if it reduces the radius
                    if np.max(np.linalg.norm(cluster_points - new_center, axis=1)) < radii[idx]:
                        centers[idx] = new_center

        return centers

    def _initialize_diverse_centers(self, X, p, rng):
        # Initialize centers as random points from X, ensuring diversity
        centers = []
        for _ in range(p):
            if not centers:
                center_idx = rng.choice(X.shape[0])
                centers.append(X[center_idx])
            else:
                max_dist = 0
                max_dist_idx = None
                for i in range(X.shape[0]):
                    min_dist = np.min(np.linalg.norm(X[i] - np.array(centers), axis=1))
                    if min_dist > max_dist:
                        max_dist = min_dist
                        max_dist_idx = i
                centers.append(X[max_dist_idx])
        return np.array(centers)

    def _find_better_medoid(self, cluster_points, current_center, rng):
        # Find a better medoid for the cluster
        num_points = cluster_points.shape[0]
        if num_points < 2:
            return None

        # Sample a subset of points for efficiency
        subset_size = min(100, num_points)
        subset_indices = rng.choice(num_points, size=subset_size, replace=False)
        subset_points = cluster_points[subset_indices]

        # Compute distances within the subset
        subset_dists = np.linalg.norm(subset_points[:, np.newaxis] - subset_points, axis=2)
        subset_radii = np.max(subset_dists, axis=1)

        # Find the point with the smallest maximum distance (medoid)
        medoid_idx = np.argmin(subset_radii)
        medoid = subset_points[medoid_idx]

        # Check if the medoid is better than the current center
        if np.max(np.linalg.norm(cluster_points - medoid, axis=1)) < np.max(np.linalg.norm(cluster_points - current_center, axis=1)):
            return np.where((cluster_points == medoid).all(axis=1))[0][0]
        else:
            return None