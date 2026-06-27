import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Initialize centers using a grid-based approach
        if rng is None:
            rng = np.random.default_rng()
        min_x, max_x = np.min(X, axis=0), np.max(X, axis=0)
        grid_size = int(np.ceil(np.power(X.shape[0], 1/3)))
        grid_points = np.meshgrid(*[np.linspace(min_x[i], max_x[i], grid_size) for i in range(X.shape[1])])
        grid_points = np.array([gp.flatten() for gp in grid_points]).T
        grid_densities = np.zeros(grid_points.shape[0])
        for i in range(grid_points.shape[0]):
            grid_densities[i] = np.sum(np.exp(-np.linalg.norm(X - grid_points[i], axis=1) ** 2 / (2 * 0.1 ** 2)))
        initial_centers_idx = np.argsort(grid_densities)[::-1][:p]
        initial_centers = grid_points[initial_centers_idx]

        # Find the closest data points to the initial centers
        centers = np.zeros((p, X.shape[1]))
        for i in range(p):
            dists = np.linalg.norm(X - initial_centers[i], axis=1)
            centers_idx = np.argmin(dists)
            if np.linalg.norm(centers - X[centers_idx], axis=1).sum() > 0:
                centers[i] = X[centers_idx]
            else:
                # If the closest data point is already a center, find the next closest one
                next_closest_idx = np.argsort(dists)[1]
                centers[i] = X[next_closest_idx]

        # Compute densities for each point
        densities = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            densities[i] = np.sum(np.exp(-np.linalg.norm(X - X[i], axis=1) ** 2 / (2 * 0.1 ** 2)))

        # Compute the local neighborhood for each point with adaptive size
        neighborhoods = []
        for i in range(X.shape[0]):
            dists = np.linalg.norm(X - X[i], axis=1)
            neighborhood_size = int(np.ceil(np.sqrt(X.shape[0]) / (1 + np.exp(-dists.sum()))))
            neighborhoods.append(np.argsort(dists)[:neighborhood_size])

        # Iterate to update centers
        for _ in range(10):
            # Compute the score for each point as a potential center
            scores = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                # Compute the density of the point's neighborhood
                neighborhood_density = np.mean([densities[j] for j in neighborhoods[i]])
                # Compute the score as the product of the point's density and its neighborhood's density
                scores[i] = densities[i] * neighborhood_density

            # Update centers
            max_score_idx = np.argmax(scores)
            if X[max_score_idx] not in centers:
                centers[np.argmin(np.linalg.norm(centers - X[max_score_idx], axis=1))] = X[max_score_idx]
            scores[max_score_idx] = -np.inf

        return centers