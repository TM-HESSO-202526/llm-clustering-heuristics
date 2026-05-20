import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        def sse(centers, X):
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            return np.sum(np.min(distances, axis=1))

        def gradient_descent(centers, X, learning_rate=0.1, iterations=100, momentum=0.9, regularization=0.01):
            velocity = np.zeros_like(centers)
            learning_rate_schedule = np.linspace(learning_rate, 0.01, iterations)
            for i in range(iterations):
                distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                assignments = np.argmin(distances, axis=1)
                for j in range(p):
                    assigned_points = X[assignments == j]
                    if len(assigned_points) > 0:
                        gradient = np.mean(assigned_points, axis=0) - centers[j] + regularization * centers[j]
                        velocity[j] = momentum * velocity[j] + learning_rate_schedule[i] * gradient
                        centers[j] += velocity[j]
            return centers

        def hierarchical_initialization(X, p, rng):
            if p == 1:
                return np.mean(X, axis=0).reshape(1, -1)
            else:
                initial_centers = np.array([np.mean(X, axis=0)])
                for _ in range(p - 1):
                    distances = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
                    farthest_points = np.argmax(np.min(distances, axis=1))
                    initial_centers = np.vstack((initial_centers, X[farthest_points]))
                return initial_centers

        initial_centers = hierarchical_initialization(X, p, rng)
        centers = gradient_descent(initial_centers, X)

        return centers