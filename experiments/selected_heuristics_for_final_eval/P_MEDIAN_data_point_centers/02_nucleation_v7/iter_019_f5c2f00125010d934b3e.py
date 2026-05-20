import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        n, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers with farthest points
        centers = [X[rng.choice(n)]]
        for _ in range(p - 1):
            dist = np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2).min(axis=1)
            next_center_idx = np.argmax(dist)
            centers.append(X[next_center_idx])

        # Convert centers list to array
        centers = np.array(centers)

        # Initialize min_dist array
        min_dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2).min(axis=1)

        for _ in range(30):  # Increase iterations for better convergence
            # Compute new potential centers
            point_weights = min_dist
            point_weights /= point_weights.sum()
            new_center_idx = rng.choice(n, p=point_weights, size=p)
            new_centers = X[new_center_idx]

            # Replace the center with the largest total distance to its points
            point_assignments = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
            cluster_sizes = np.bincount(point_assignments, minlength=p)
            cluster_distances = np.sum(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=0)
            center_costs = cluster_distances / cluster_sizes
            center_replaced = np.argmax(center_costs)
            new_center_costs = np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2).min(axis=1)
            new_center_costs = np.sum(new_center_costs)
            old_center_costs = np.sum(min_dist)
            if new_center_costs < old_center_costs:
                centers[center_replaced] = new_centers[np.argmin(np.linalg.norm(centers - new_centers, axis=1))]

            # Update min_dist array
            min_dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2).min(axis=1)

            # Introduce a local search to improve the solution
            for _ in range(5):
                point_assignments = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                cluster_sizes = np.bincount(point_assignments, minlength=p)
                for i in range(p):
                    cluster_points = X[point_assignments == i]
                    if cluster_points.size > 0:
                        new_center = np.mean(cluster_points, axis=0)
                        new_center_idx = np.argmin(np.linalg.norm(X - new_center, axis=1))
                        new_center = X[new_center_idx]
                        if np.any(np.linalg.norm(centers - new_center, axis=1) > 1e-6):
                            centers[i] = new_center

        # Perform an additional local search to refine the solution
        for _ in range(10):
            point_assignments = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
            cluster_sizes = np.bincount(point_assignments, minlength=p)
            for i in range(p):
                cluster_points = X[point_assignments == i]
                if cluster_points.size > 0:
                    new_center = np.mean(cluster_points, axis=0)
                    new_center_idx = np.argmin(np.linalg.norm(X - new_center, axis=1))
                    new_center = X[new_center_idx]
                    if np.any(np.linalg.norm(centers - new_center, axis=1) > 1e-6):
                        centers[i] = new_center

        return centers