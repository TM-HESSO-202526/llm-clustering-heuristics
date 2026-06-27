import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Intended mechanism family: Recursive partition clustering
        n, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers using a recursive partitioning approach
        def recursive_partition(points, num_centers):
            if num_centers == 1:
                return np.mean(points, axis=0, keepdims=True)
            else:
                # Split the points along the longest dimension
                longest_dim = np.argmax(np.max(points, axis=0) - np.min(points, axis=0))
                split_point = np.median(points[:, longest_dim])
                left_points = points[points[:, longest_dim] < split_point]
                right_points = points[points[:, longest_dim] >= split_point]

                # Recursively partition the left and right points
                left_centers = recursive_partition(left_points, num_centers // 2)
                right_centers = recursive_partition(right_points, num_centers - num_centers // 2)

                return np.concatenate((left_centers, right_centers), axis=0)

        centers = recursive_partition(X, p)

        # Refine the centers using a bounded SSE-compatible refinement step
        for _ in range(5):
            # Assign each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update the centers as the mean of the assigned points
            new_centers = np.array([np.mean(X[assignments == i], axis=0) if np.any(assignments == i) else centers[i] for i in range(p)])

            # Check for convergence
            if np.all(new_centers == centers):
                break

            centers = new_centers

        return centers