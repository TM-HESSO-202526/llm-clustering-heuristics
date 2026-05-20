import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        """
        Returns p centers from a uniform random sample S of size min(n_full, 10*p).

        Args:
        - X (numpy array of shape (m, d)): Sample S.
        - p (int): Number of centers.
        - rng (optional numpy.random.Generator): Random number generator.

        Returns:
        - centers (numpy array of shape (p, d)): p centers in R^d.
        """
        # If rng is None, create a default random number generator
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers with random points from the sample using a variation of k-means++
        centers = X[rng.choice(X.shape[0], size=1, replace=False)]
        for _ in range(1, p):
            dist2 = np.array([np.min(np.linalg.norm(x - centers, axis=1)) ** 2 for x in X])
            dist2 = np.maximum(dist2, np.percentile(dist2, 25))  # Avoids extreme outliers
            probs = dist2 / dist2.sum()
            cumulative_probs = np.cumsum(probs)
            r = rng.random()
            ind = np.where(cumulative_probs >= r)[0][0]
            centers = np.vstack((centers, X[ind]))

        # Repeat the clustering process for a fixed number of iterations with adaptive convergence
        for _ in range(10):
            # Assign each point in the sample to the closest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Update centers as the weighted mean of all points assigned to each center
            new_centers = np.array([np.average(X[labels == i], axis=0) if np.any(labels == i) else centers[i] for i in range(p)])

            # Check for adaptive convergence
            if np.allclose(centers, new_centers, atol=1e-4 * np.std(X, axis=0)):
                break

            # Introduce a small perturbation to the new centers to avoid local minima
            new_centers += rng.normal(loc=0, scale=0.01 * np.std(X, axis=0), size=new_centers.shape)

            centers = new_centers

        return centers