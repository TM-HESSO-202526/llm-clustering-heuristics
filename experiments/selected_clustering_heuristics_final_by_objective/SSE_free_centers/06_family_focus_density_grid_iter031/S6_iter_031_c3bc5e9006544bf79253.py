import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Determine the number of bins for each dimension based on the number of points and dimensions
        n_bins = int(np.ceil(np.log2(X.shape[0] / X.shape[1])))

        # Initialize a grid with the specified number of bins
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        bins = np.array([np.linspace(min_val, max_val, n_bins) for min_val, max_val in zip(min_vals, max_vals)])

        # Initialize a density summary for each cell in the grid
        density = np.zeros(tuple([n_bins]*X.shape[1]))

        # Assign each point to a cell in the grid and update the density
        for point in X:
            idx = [int(np.searchsorted(bin, coord)) - 1 for bin, coord in zip(bins, point)]
            density[tuple(idx)] += 1

        # Normalize the density by the total number of points
        density /= X.shape[0]

        # Select the top p cells with the highest density
        idx = np.unravel_index(np.argsort(-density, axis=None)[:p], density.shape)

        # Initialize the centers at the centroid of the selected cells
        centers = np.zeros((p, X.shape[1]))
        for i, idx_tuple in enumerate(zip(*idx)):
            cell_mins = [bins[d][idx] for d, idx in zip(range(X.shape[1]), idx_tuple)]
            cell_maxs = [bins[d][idx + 1] for d, idx in zip(range(X.shape[1]), idx_tuple)]
            cell_centers = [(cell_min + cell_max) / 2 for cell_min, cell_max in zip(cell_mins, cell_maxs)]
            centers[i] = cell_centers

        # Perform bounded Lloyd-style refinement with a convergence check
        for _ in range(20):
            # Assign each point to the nearest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Update the centers to the centroid of the assigned points
            new_centers = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(p)])

            # Check for convergence
            if np.allclose(centers, new_centers, atol=1e-6):
                break

            centers = new_centers

        # Perform one final refinement step with a smaller convergence threshold
        for _ in range(5):
            # Assign each point to the nearest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

            # Update the centers to the weighted centroid of the assigned points
            weights = np.array([np.sum(labels == i) / X.shape[0] for i in range(p)])
            new_centers = np.array([np.average(X[labels == i], axis=0, weights=np.ones(X[labels == i].shape[0])) if np.any(labels == i) else centers[i] for i in range(p)])

            # Check for convergence
            if np.allclose(centers, new_centers, atol=1e-8):
                break

            centers = new_centers

            # Randomly perturb centers to avoid local minima
            if rng is not None:
                centers += 0.01 * (rng.random(size=centers.shape) - 0.5)

        return centers