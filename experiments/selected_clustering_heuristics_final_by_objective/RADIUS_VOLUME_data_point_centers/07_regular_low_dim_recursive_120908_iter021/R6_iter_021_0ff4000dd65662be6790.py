import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        n, d = X.shape
        
        # Initial centers are chosen randomly from data points
        centers = X[rng.choice(n, size=p, replace=False)]
        
        for _ in range(100):  # Perform 100 iterations of refinement
            # Assign each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Compute the radius of each cluster
            radii = np.zeros(p)
            for i in range(p):
                cluster_points = X[assignments == i]
                if cluster_points.size > 0:
                    radii[i] = np.max(np.linalg.norm(cluster_points - centers[i], axis=1))
            
            # Identify the cluster with the largest radius
            max_radius_index = np.argmax(radii)
            
            # Split the cluster with the largest radius into two new clusters
            if radii[max_radius_index] > 0:
                cluster_points = X[assignments == max_radius_index]
                distances_to_center = np.linalg.norm(cluster_points - centers[max_radius_index], axis=1)
                farthest_point_index = np.argmax(distances_to_center)
                farthest_point = cluster_points[farthest_point_index]
                
                # Replace the old center with the point that is farthest from the center
                # and has the minimum radius
                min_radius = np.inf
                best_point = None
                for point in cluster_points:
                    distances_to_point = np.linalg.norm(cluster_points - point, axis=1)
                    radius = np.max(distances_to_point)
                    if radius < min_radius:
                        min_radius = radius
                        best_point = point
                
                # Replace the old center with the new center
                new_center = best_point
                
                # Check if the new center is too close to another center
                too_close = False
                for i in range(p):
                    if i != max_radius_index and np.linalg.norm(new_center - centers[i]) < radii[i] / 2:
                        too_close = True
                        break
                
                if not too_close:
                    centers[max_radius_index] = new_center
                
                else:
                    # Replace the old center with the point that is farthest from all other centers
                    distances_to_other_centers = np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                    farthest_point = X[np.argmax(distances_to_other_centers)]
                    centers[max_radius_index] = farthest_point
            
            # Merge two clusters with the smallest radii
            if p > 1:
                min_radius_indices = np.argsort(radii)[:2]
                if radii[min_radius_indices[0]] < radii[min_radius_indices[1]] / 2:
                    # Merge the two clusters
                    centers[min_radius_indices[0]] = X[np.argmin(np.linalg.norm(X - np.mean(centers[min_radius_indices], axis=0), axis=1))]
                    centers = np.delete(centers, min_radius_indices[1], axis=0)
                    p -= 1
        
        # Refine the centers by moving them to the point with the minimum radius
        for _ in range(10):
            for i in range(p):
                cluster_points = X[np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1) == i]
                if cluster_points.size > 0:
                    distances_to_center = np.linalg.norm(cluster_points - centers[i], axis=1)
                    min_radius_index = np.argmin(distances_to_center)
                    min_radius_point = cluster_points[min_radius_index]
                    centers[i] = min_radius_point
        
        # Ensure p centers
        if p < len(centers):
            centers = centers[:p]
        elif p > len(centers):
            remaining_points = X[~np.isin(X, centers).all(axis=1)]
            if len(remaining_points) > 0:
                new_centers = remaining_points[:p - len(centers)]
                centers = np.vstack((centers, new_centers))
        
        # Additional refinement step to improve the radius of the clusters
        for _ in range(10):
            for i in range(p):
                cluster_points = X[np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1) == i]
                if cluster_points.size > 0:
                    distances_to_center = np.linalg.norm(cluster_points - centers[i], axis=1)
                    max_distance_index = np.argmax(distances_to_center)
                    max_distance_point = cluster_points[max_distance_index]
                    new_center = X[np.argmin(np.linalg.norm(X - (centers[i] + max_distance_point) / 2, axis=1))]
                    centers[i] = new_center
        
        # New refinement step: try to replace each center with a point that minimizes the radius
        for _ in range(10):
            for i in range(p):
                cluster_points = X[np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1) == i]
                if cluster_points.size > 0:
                    min_radius = np.inf
                    best_point = None
                    for point in cluster_points:
                        distances_to_point = np.linalg.norm(cluster_points - point, axis=1)
                        radius = np.max(distances_to_point)
                        if radius < min_radius:
                            min_radius = radius
                            best_point = point
                    centers[i] = best_point
        
        return centers