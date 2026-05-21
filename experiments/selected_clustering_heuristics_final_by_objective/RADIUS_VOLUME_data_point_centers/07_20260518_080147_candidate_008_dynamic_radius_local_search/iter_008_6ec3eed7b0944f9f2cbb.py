import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n, d = X.shape

        # Initialize centers randomly
        centers = X[rng.choice(n, size=p, replace=False)]

        def calculate_radius(X, centers):
            # Assign points to centers
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Calculate radius for each center
            radii = np.zeros(p)
            points_in_clusters = [X[assignments == i] for i in range(p)]
            for i in range(p):
                if len(points_in_clusters[i]) > 0:
                    radii[i] = np.max(np.linalg.norm(points_in_clusters[i] - centers[i], axis=1))
            return radii, assignments, points_in_clusters

        def calculate_objective(radii, d):
            return np.sum(radii ** d)

        def get_worst_center(radii):
            return np.argmax(radii)

        def get_best_point(points_in_cluster):
            # Select a point that minimizes the maximum distance to all other points
            best_point_index = 0
            min_max_distance = float('inf')
            for i in range(len(points_in_cluster)):
                max_distance = np.max(np.linalg.norm(points_in_cluster[i] - points_in_cluster, axis=1))
                if max_distance < min_max_distance:
                    min_max_distance = max_distance
                    best_point_index = i
            return points_in_cluster[best_point_index]

        def local_search(centers, X, p, d):
            radii, assignments, points_in_clusters = calculate_radius(X, centers)
            for _ in range(10):
                # Try to move each center to a nearby point
                for i in range(p):
                    points_in_cluster = points_in_clusters[i]
                    if len(points_in_cluster) > 0:
                        # Try to move the center to a nearby point
                        new_center = get_best_point(points_in_cluster)
                        new_centers = centers.copy()
                        new_centers[i] = new_center
                        new_radii, new_assignments, new_points_in_clusters = calculate_radius(X, new_centers)
                        new_objective = calculate_objective(new_radii, d)
                        if new_objective < calculate_objective(radii, d):
                            centers = new_centers
                            radii = new_radii
                            points_in_clusters = new_points_in_clusters
            return centers

        def dynamic_radius_control(centers, X, p, d):
            radii, assignments, points_in_clusters = calculate_radius(X, centers)
            for _ in range(5):
                worst_center_index = get_worst_center(radii)
                points_in_worst_cluster = points_in_clusters[worst_center_index]
                if len(points_in_worst_cluster) > 0:
                    # Calculate the median point in the worst cluster
                    median_point = np.median(points_in_worst_cluster, axis=0)
                    new_centers = centers.copy()
                    new_centers[worst_center_index] = median_point
                    new_radii, new_assignments, new_points_in_clusters = calculate_radius(X, new_centers)
                    new_objective = calculate_objective(new_radii, d)
                    if new_objective < calculate_objective(radii, d):
                        centers = new_centers
                        radii = new_radii
                        points_in_clusters = new_points_in_clusters
            return centers

        # Repeat until no improvement
        previous_objective = float('inf')
        for _ in range(10):
            radii, assignments, points_in_clusters = calculate_radius(X, centers)
            objective = calculate_objective(radii, d)
            if objective >= previous_objective:
                break
            previous_objective = objective

            # Replace the worst center
            worst_center_index = get_worst_center(radii)
            points_in_worst_cluster = points_in_clusters[worst_center_index]
            if len(points_in_worst_cluster) > 0:
                new_center = get_best_point(points_in_worst_cluster)
                centers[worst_center_index] = new_center

            # Perform local search
            centers = local_search(centers, X, p, d)

            # Apply dynamic radius control
            centers = dynamic_radius_control(centers, X, p, d)

        return centers