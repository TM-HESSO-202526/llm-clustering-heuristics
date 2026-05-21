import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize p centers randomly from data points
        centers = X[rng.choice(X.shape[0], size=p, replace=False)]

        for _ in range(100):  # max iterations
            # Assign each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Compute the radius for each cluster
            radii = np.array([np.max(distances[labels == i, i]) if np.sum(labels == i) > 0 else 0 for i in range(p)])

            # Identify high-radius clusters and replace their centers with a data point that minimizes the radius
            for cluster in np.argsort(-radii)[:int(p / 2)]:
                points_in_cluster = X[labels == cluster]
                if points_in_cluster.shape[0] > 1:
                    # Initialize the minimum radius and the corresponding center
                    min_radius = np.inf
                    new_center = None

                    # Try each data point in the cluster as the new center
                    for point in points_in_cluster:
                        # Compute the radius if the point is the center
                        distances_to_point = np.linalg.norm(points_in_cluster - point, axis=1)
                        radius = np.max(distances_to_point)

                        # Update the minimum radius and the corresponding center
                        if radius < min_radius:
                            min_radius = radius
                            new_center = point

                    # Replace the center with the new center
                    centers[cluster] = new_center

            # Dynamically adjust the center positions to minimize the radius
            for i in range(p):
                points_in_cluster = X[labels == i]
                if points_in_cluster.shape[0] > 1:
                    # Initialize the minimum radius and the corresponding center
                    min_radius = np.inf
                    new_center = None

                    # Try each data point in the cluster as the new center
                    for point in points_in_cluster:
                        # Compute the radius if the point is the center
                        distances_to_point = np.linalg.norm(points_in_cluster - point, axis=1)
                        radius = np.max(distances_to_point)

                        # Update the minimum radius and the corresponding center
                        if radius < min_radius:
                            min_radius = radius
                            new_center = point

                    # Replace the center with the new center
                    centers[i] = new_center

            # Check for empty clusters and replace their centers
            for i in range(p):
                if np.sum(labels == i) == 0:
                    # Replace the center with a random data point that is farthest from existing centers
                    distances_to_centers = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                    distances_to_centers = np.min(distances_to_centers, axis=1)
                    farthest_point_index = np.argmax(distances_to_centers)
                    centers[i] = X[farthest_point_index]

            # Split high-radius clusters into smaller clusters
            for i in range(p):
                points_in_cluster = X[labels == i]
                if points_in_cluster.shape[0] > 1:
                    # Compute the radius of the cluster
                    radius = np.max(np.linalg.norm(points_in_cluster - centers[i], axis=1))

                    # If the radius is large, split the cluster into two smaller clusters
                    if radius > np.mean(radii) * 2:
                        # Select the two points that are farthest from each other as the new centers
                        distances = np.linalg.norm(points_in_cluster[:, np.newaxis] - points_in_cluster, axis=2)
                        distances_to_self = np.eye(points_in_cluster.shape[0]) * np.inf
                        distances += distances_to_self
                        point1, point2 = np.unravel_index(np.argmax(distances), distances.shape)
                        new_center1 = points_in_cluster[point1]
                        new_center2 = points_in_cluster[point2]

                        # Replace the center with the new center
                        centers[i] = new_center1
                        centers = np.vstack((centers, new_center2))

                        # Assign each point to the nearest center
                        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                        labels = np.argmin(distances, axis=1)

                        # Remove the empty clusters
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        if np.any(counts == 0):
                            empty_clusters = unique_labels[counts == 0]
                            centers = np.delete(centers, empty_clusters, axis=0)

        # Ensure the final centers are data points
        for i in range(p):
            distances_to_data_points = np.linalg.norm(centers[i] - X, axis=1)
            centers[i] = X[np.argmin(distances_to_data_points)]

        return centers[:p]