import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        m, d = X.shape
        if p > m:
            # Sample points with replacement to create p centers
            idx = rng.choice(m, size=p, replace=True)
            return X[idx]

        # Initialize centers by choosing random points from the sample
        # with a higher probability for points farther away from the existing centers
        centers = []
        for _ in range(p):
            if not centers:
                idx = rng.choice(m)
                centers.append(X[idx])
            else:
                centers_array = np.array(centers)
                dist = np.linalg.norm(X[:, np.newaxis] - centers_array, axis=2) ** 2
                dist = np.min(dist, axis=1)
                probabilities = dist / np.sum(dist)
                idx = rng.choice(m, p=probabilities)
                centers.append(X[idx])

        # Perform a small bounded refinement on the sample
        for _ in range(15):  # Increased iterations for convergence
            # Assign each point to the nearest center
            dist = np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2) ** 2
            labels = np.argmin(dist, axis=1)

            # Compute the mean of each cluster
            new_centers = []
            for i in range(p):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    # If a cluster is empty, reinitialize the center
                    idx = rng.choice(m)
                    new_centers.append(X[idx])

            # Stop if centers do not change
            if np.allclose(np.array(centers), np.array(new_centers)):
                break

            centers = new_centers

        # Additional step: perturb the centers slightly to escape local minima
        for i in range(p):
            centers[i] += rng.normal(0, 0.1, size=d)

        return np.array(centers)