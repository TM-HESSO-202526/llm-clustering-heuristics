import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers using K-Means++ strategy
        centers = np.array([X[rng.integers(X.shape[0])]])
        for _ in range(1, p):
            dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            min_dist = np.min(dist, axis=1)
            probs = min_dist / np.sum(min_dist)
            idx = rng.choice(X.shape[0], p=probs)
            centers = np.vstack((centers, X[idx]))

        # Perform a few iterations of K-Means to refine centers
        for _ in range(5):
            # Assign each point to the nearest center
            dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(dist, axis=1)

            # Update centers as the mean of all points assigned to each center
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(p) if np.any(labels == i)])
            if new_centers.size == 0:
                break
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        # Minimize intra-cluster dispersion by reassigning points to the nearest center
        for _ in range(2):
            dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(dist, axis=1)
            for i in range(p):
                points_in_cluster = X[labels == i]
                new_center = np.mean(points_in_cluster, axis=0) if points_in_cluster.size > 0 else centers[i]
                centers[i] = new_center

        # Randomized center refinement
        for _ in range(3):
            for i in range(p):
                labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                points_in_cluster = X[labels == i]
                if points_in_cluster.size > 0:
                    new_center = np.mean(points_in_cluster + 0.1 * rng.normal(0, 1, size=points_in_cluster.shape), axis=0)
                    centers[i] = new_center

        # Reduce intra-cluster radius by moving the center towards the centroid of the furthest points
        for _ in range(3):
            for i in range(p):
                labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                points_in_cluster = X[labels == i]
                if points_in_cluster.size > 0:
                    dist_to_center = np.linalg.norm(points_in_cluster - centers[i], axis=1)
                    furthest_points = points_in_cluster[dist_to_center >= np.mean(dist_to_center)]
                    if furthest_points.size > 0:
                        new_center = centers[i] + 0.1 * (np.mean(furthest_points, axis=0) - centers[i])
                        centers[i] = new_center

        return centers