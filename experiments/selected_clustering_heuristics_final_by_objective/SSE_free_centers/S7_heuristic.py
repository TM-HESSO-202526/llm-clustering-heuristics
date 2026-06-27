import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Intended mechanism family: Clustering via local density peaks
        # This method identifies clusters as regions of high density in the data

        if rng is None:
            rng = np.random.default_rng()

        n, d = X.shape

        # Compute the distance matrix
        distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

        # Compute the local density of each point
        densities = np.sum(np.exp(-((distances / np.mean(distances)) ** 2)), axis=1)

        # Compute the minimum distance to a higher-density point
        delta = np.inf * np.ones(n)
        for i in range(n):
            for j in range(n):
                if densities[j] > densities[i]:
                    delta[i] = min(delta[i], distances[i, j])

        # Identify the points with the highest density and minimum distance to a higher-density point
        peak_indices = np.argsort(densities * delta)[::-1]

        # Select the top p peak indices as the initial centers
        centers = X[peak_indices[:p]]

        # Perform a final refinement step to minimize the sum of squared Euclidean distances
        for _ in range(10):  # Perform 10 iterations of refinement
            # Assign each point to the nearest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Compute new centers as the mean of points in each cluster
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(p)])

            # Check for convergence
            if np.all(centers == new_centers):
                break

            centers = new_centers

        return centers