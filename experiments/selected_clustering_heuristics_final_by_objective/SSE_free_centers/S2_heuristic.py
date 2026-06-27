import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Adaptive initialization with k-means++ and random sampling
        centers = self._adaptive_init(X, p, rng)

        # Hybrid annealing loop with noise injection and iterative refinement
        for t in range(200):
            temperature = 1 - t / 200

            # Calculate distances and assignments
            distances = self._calculate_distances(X, centers)
            assignments = np.argmin(distances, axis=1)

            # Update centers with noise injection and iterative refinement
            new_centers = np.copy(centers)
            for i in range(p):
                points_in_cluster = X[assignments == i]
                if len(points_in_cluster) > 0:
                    mean = np.mean(points_in_cluster, axis=0)
                    noise = temperature * np.sqrt(np.mean(np.linalg.norm(points_in_cluster - mean, axis=1)**2)) * rng.uniform(-1, 1, size=mean.shape)
                    new_centers[i] = mean + noise

            # Check for convergence
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

            # Iterative refinement with adaptive convergence check
            if t % 10 == 0:  # Reduced interval for more frequent refinements
                for _ in range(5):
                    new_centers = np.copy(centers)
                    for i in range(p):
                        points_in_cluster = X[assignments == i]
                        if len(points_in_cluster) > 0:
                            mean = np.mean(points_in_cluster, axis=0)
                            new_centers[i] = mean
                    distances = self._calculate_distances(X, new_centers)
                    new_assignments = np.argmin(distances, axis=1)
                    if np.array_equal(assignments, new_assignments):
                        break
                    assignments = new_assignments
                    centers = new_centers

        # Final local search with improved convergence check
        for _ in range(20):  # Increased number of local search iterations
            new_centers = np.copy(centers)
            for i in range(p):
                points_in_cluster = X[assignments == i]
                if len(points_in_cluster) > 0:
                    mean = np.mean(points_in_cluster, axis=0)
                    new_centers[i] = mean
            distances = self._calculate_distances(X, new_centers)
            new_assignments = np.argmin(distances, axis=1)
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments
            centers = new_centers

        return centers

    def _adaptive_init(self, X, p, rng):
        centers = np.zeros((p, X.shape[1]))
        if p > X.shape[0]:
            centers[:X.shape[0]] = X
            for i in range(X.shape[0], p):
                centers[i] = rng.uniform(X.min(axis=0), X.max(axis=0))
        else:
            # K-Means++ initialization with improved seeding
            centers[0] = X[rng.choice(X.shape[0], size=1, replace=False)].flatten()

            for i in range(1, p):
                distances = np.linalg.norm(X[:, np.newaxis] - centers[:i], axis=2)
                min_distances = np.min(distances, axis=1)
                probabilities = min_distances / np.sum(min_distances)
                next_center_index = rng.choice(X.shape[0], size=1, p=probabilities)
                centers[i] = X[next_center_index].flatten()

        return centers

    def _calculate_distances(self, X, centers):
        return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)