import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        """
        Enhanced Sample-Based p-Median Heuristic with Iterative Farthest Point Clustering and Voronoi Refining.

        Parameters:
        X (numpy array): Sample S of shape (m, d)
        p (int): Number of centers
        rng (numpy.random.Generator, optional): Random number generator

        Returns:
        centers (numpy array): p centers of shape (p, d)
        """
        m, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Initialize with a random center
        centers = np.array([X[rng.integers(m)]])
        min_dist = np.linalg.norm(X - centers[0], axis=1)

        # Greedily add centers until p is reached
        for _ in range(1, p):
            # Calculate distances from each point to its nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            min_dist = np.min(distances, axis=1)

            # Select the point with the maximum minimum distance as the next center
            next_center_idx = np.argmax(min_dist)
            centers = np.vstack((centers, X[next_center_idx]))

            # Update min_dist to ensure the newly added center is considered
            new_dist = np.linalg.norm(X - centers[-1], axis=1)
            min_dist = np.minimum(min_dist, new_dist)

        # Iterative Voronoi refining
        for _ in range(5 * p):
            # Assign each point to its nearest center
            assignments = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Update centers as the mean of their assigned points
            new_centers = np.array([X[assignments == i].mean(axis=0) if np.any(assignments == i) else centers[i] for i in range(p)])

            # Check for convergence
            if np.all(np.linalg.norm(centers - new_centers, axis=1) < 1e-6):
                break

            centers = new_centers

        # Select p points from X that are closest to the refined centers
        final_centers = np.array([X[np.argmin(np.linalg.norm(X - center, axis=1))] for center in centers])

        return final_centers