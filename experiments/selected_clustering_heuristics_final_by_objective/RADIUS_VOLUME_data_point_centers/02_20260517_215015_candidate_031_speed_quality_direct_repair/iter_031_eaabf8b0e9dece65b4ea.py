import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize centers with k-means++ like initialization
        centers = X[rng.choice(X.shape[0], size=1, replace=False)]
        for _ in range(p - 1):
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            new_center_index = rng.choice(X.shape[0], p=probabilities)
            centers = np.vstack((centers, X[new_center_index]))

        for _ in range(250):  # Increased iteration limit
            # Assign each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Compute the radius of each cluster
            radii = np.zeros(p)
            for j in range(p):
                points_in_cluster = X[labels == j]
                if points_in_cluster.size > 0:
                    radii[j] = np.max(np.linalg.norm(points_in_cluster - centers[j], axis=1))

            # Identify the cluster with the largest radius^d contribution
            d = X.shape[1]
            radius_contributions = radii ** d
            max_contribution_indices = np.argsort(radius_contributions)[::-1]

            # Replace the center of the cluster with the largest radius^d contribution
            for max_contribution_index in max_contribution_indices[:15]:  # Replace top 15 clusters
                points_in_max_cluster = X[labels == max_contribution_index]
                if points_in_max_cluster.size > 0:
                    # Local search to find a better center
                    best_center_index = -1
                    best_radius = np.inf
                    for i in range(points_in_max_cluster.shape[0]):
                        new_center = points_in_max_cluster[i]
                        new_distances = np.linalg.norm(points_in_max_cluster - new_center, axis=1)
                        new_radius = np.max(new_distances)
                        if new_radius < best_radius:
                            best_radius = new_radius
                            best_center_index = i
                    if best_center_index != -1:
                        centers[max_contribution_index] = points_in_max_cluster[best_center_index]

            # Neighborhood exploration to further improve the centers
            for _ in range(20):  # Increased neighborhood exploration iterations
                i, j = rng.choice(p, size=2, replace=False)
                points_in_i = X[labels == i]
                points_in_j = X[labels == j]
                if points_in_i.size > 0 and points_in_j.size > 0:
                    new_center_i = points_in_i[rng.choice(points_in_i.shape[0])]
                    new_center_j = points_in_j[rng.choice(points_in_j.shape[0])]
                    new_distances_i = np.linalg.norm(points_in_i - new_center_i, axis=1)
                    new_radius_i = np.max(new_distances_i)
                    new_distances_j = np.linalg.norm(points_in_j - new_center_j, axis=1)
                    new_radius_j = np.max(new_distances_j)
                    if new_radius_i < radii[i] and new_radius_j < radii[j]:
                        centers[i] = new_center_i
                        centers[j] = new_center_j
                        radii[i] = new_radius_i
                        radii[j] = new_radius_j

            # Improved radius reduction
            for _ in range(25):  # Increased radius reduction iterations
                for i in range(p):
                    points_in_i = X[labels == i]
                    if points_in_i.size > 0:
                        new_center = np.mean(points_in_i, axis=0)
                        new_distances_i = np.linalg.norm(points_in_i - new_center, axis=1)
                        new_radius_i = np.max(new_distances_i)
                        if new_radius_i < radii[i]:
                            centers[i] = new_center
                            radii[i] = new_radius_i

            # Cluster splitting
            for i in range(p):
                points_in_i = X[labels == i]
                if points_in_i.size > 0:
                    # Find the farthest point from the center
                    farthest_point = points_in_i[np.argmax(np.linalg.norm(points_in_i - centers[i], axis=1))]
                    # Split the cluster into two
                    new_center = farthest_point
                    new_distances_i = np.linalg.norm(points_in_i - new_center, axis=1)
                    new_radius_i = np.max(new_distances_i)
                    if new_radius_i < radii[i] * 0.8:
                        centers = np.vstack((centers, new_center))
                        radii = np.append(radii, new_radius_i)
                        p += 1
                        # Reassign points to the new cluster
                        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                        labels = np.argmin(distances, axis=1)
                        # Update radii
                        radii = np.zeros(p)
                        for j in range(p):
                            points_in_cluster = X[labels == j]
                            if points_in_cluster.size > 0:
                                radii[j] = np.max(np.linalg.norm(points_in_cluster - centers[j], axis=1))
                        break

        # Remove excess centers
        while p > len(np.unique(labels)):
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)
            counts = np.bincount(labels, minlength=p)
            min_count_index = np.argmin(counts)
            centers = np.delete(centers, min_count_index, axis=0)
            p -= 1

        return centers