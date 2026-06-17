import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize p centers randomly from data points
        centers = X[rng.choice(X.shape[0], size=p, replace=False)]

        # Repeat high-radius cluster splitting and repair until convergence
        for _ in range(10):  # Max iterations
            # Assign each point to its nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            nearest_center = np.argmin(distances, axis=1)

            # Calculate cluster radii
            cluster_radii = np.array([np.max(distances[nearest_center == i, i]) for i in range(p)])

            # Identify the cluster with the largest radius^d contribution
            largest_radius_cluster_idx = np.argmax(cluster_radii ** X.shape[1])

            # Calculate the radius^d contribution of the largest cluster
            largest_radius_contribution = np.max(cluster_radii ** X.shape[1])

            # If the largest cluster radius is too large, split it
            if largest_radius_contribution > 0.5 * np.sum(cluster_radii ** X.shape[1]):
                # Choose the point in the largest cluster that is farthest from the center
                farthest_point_idx = np.argmax(distances[nearest_center == largest_radius_cluster_idx, largest_radius_cluster_idx])

                # Replace the center of the largest cluster with the farthest point
                new_centers = [centers[largest_radius_cluster_idx], X[farthest_point_idx]]

                # Remove the farthest point from the cluster
                points_in_largest_cluster = np.where(nearest_center == largest_radius_cluster_idx)[0]
                points_in_largest_cluster = np.delete(points_in_largest_cluster, np.where(points_in_largest_cluster == farthest_point_idx)[0])

                # Assign points to the new centers
                new_distances = np.linalg.norm(X[points_in_largest_cluster][:, np.newaxis] - new_centers, axis=2)
                new_nearest_center = np.argmin(new_distances, axis=1)

                # Refine the new centers
                new_cluster_radii = np.array([np.max(new_distances[new_nearest_center == i, i]) for i in range(2)])
                if new_cluster_radii[0] < new_cluster_radii[1]:
                    centers[largest_radius_cluster_idx] = new_centers[0]
                else:
                    centers[largest_radius_cluster_idx] = new_centers[1]

                # Add the other new center to the list of centers
                new_center = new_centers[0] if new_cluster_radii[0] < new_cluster_radii[1] else new_centers[1]
                if len(centers) < p:
                    centers = np.vstack((centers, new_center))
                else:
                    # Remove the center with the smallest cluster size
                    cluster_sizes = np.array([np.sum(nearest_center == i) for i in range(p)])
                    smallest_cluster_idx = np.argmin(cluster_sizes)
                    centers = np.delete(centers, smallest_cluster_idx, axis=0)
                    centers = np.vstack((centers, new_center))

            # Otherwise, refine the centers using bounded medoid replacement
            else:
                for i in range(p):
                    # Choose a few random points in the cluster as replacement centers
                    points_in_cluster = np.where(nearest_center == i)[0]
                    replacement_center_idxs = rng.choice(points_in_cluster, size=min(10, len(points_in_cluster)), replace=False)

                    # Calculate the new cluster radius for each replacement center
                    new_cluster_radii = np.array([np.max(np.linalg.norm(X[points_in_cluster] - X[j], axis=1)) for j in replacement_center_idxs])

                    # If a replacement center results in a smaller cluster radius, replace the center
                    if np.any(new_cluster_radii < cluster_radii[i]):
                        best_replacement_idx = replacement_center_idxs[np.argmin(new_cluster_radii)]
                        centers[i] = X[best_replacement_idx]

        # Ensure the final centers are data points
        centers = np.array([X[np.argmin(np.linalg.norm(X - center, axis=1))] for center in centers])

        return centers