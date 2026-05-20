import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        def score(centers):
            """Evaluate the score of a center set"""
            n = X.shape[0]
            d = X.shape[1]
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            nearest_center = np.argmin(distances, axis=1)
            cluster_radii = np.zeros(centers.shape[0])
            for i in range(centers.shape[0]):
                points_in_cluster = X[nearest_center == i]
                if points_in_cluster.size == 0:
                    cluster_radii[i] = 0
                else:
                    cluster_radii[i] = np.max(np.linalg.norm(points_in_cluster - centers[i], axis=1))
            return np.sum(cluster_radii ** d)

        def adaptive_recursive_cover(X, p, depth=0):
            """Adaptively cover the points with spheres"""
            if p == 0 or X.size == 0:
                return np.zeros((0, X.shape[1]))
            elif p == 1:
                centroid = np.mean(X, axis=0)
                return np.array([centroid])
            else:
                # Randomly select a point as the initial center
                init_center_idx = rng.integers(X.shape[0])
                init_center = X[init_center_idx]
                
                # Find the point farthest from the initial center
                farthest_point_idx = np.argmax(np.linalg.norm(X - init_center, axis=1))
                farthest_point = X[farthest_point_idx]
                
                # Assign points to the two centers
                distances = np.linalg.norm(X - init_center, axis=1)
                points_near_init_center = X[distances <= np.linalg.norm(farthest_point - init_center)]
                points_near_farthest_point = X[distances > np.linalg.norm(farthest_point - init_center)]
                
                # Recursively cover the two sets of points
                center1 = adaptive_recursive_cover(points_near_init_center, max(p // 2, 1), depth + 1)
                center2 = adaptive_recursive_cover(points_near_farthest_point, max(p - p // 2, 1), depth + 1)
                
                # Combine the two sets of centers
                centers = np.vstack((center1, center2))
                
                # Refine the centers
                for _ in range(10):
                    distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                    nearest_center = np.argmin(distances, axis=1)
                    new_centers = np.copy(centers)
                    for i in range(centers.shape[0]):
                        points_in_cluster = X[nearest_center == i]
                        if points_in_cluster.size == 0:
                            new_centers[i] = np.mean(X, axis=0)
                        else:
                            new_centers[i] = np.mean(points_in_cluster, axis=0) + 0.2 * (np.mean(points_in_cluster, axis=0) - centers[i])
                    centers = new_centers
                
                # Remove empty clusters
                nearest_center = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                unique_centers = []
                for i in range(centers.shape[0]):
                    if np.any(nearest_center == i):
                        unique_centers.append(centers[i])
                centers = np.array(unique_centers)
                
                # If the number of unique centers is less than p, add new centers by diversifying
                while centers.shape[0] < p:
                    # Select a point farthest from the existing centers
                    distances_to_centers = np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
                    farthest_point_idx = np.argmax(distances_to_centers)
                    new_center = X[farthest_point_idx]
                    centers = np.vstack((centers, new_center))
                    # Perform local optimization for the new center
                    for _ in range(5):
                        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                        nearest_center = np.argmin(distances, axis=1)
                        new_centers = np.copy(centers)
                        for i in range(centers.shape[0]):
                            points_in_cluster = X[nearest_center == i]
                            if points_in_cluster.size == 0:
                                new_centers[i] = np.mean(X, axis=0)
                            else:
                                new_centers[i] = np.mean(points_in_cluster, axis=0) + 0.1 * (np.mean(points_in_cluster, axis=0) - centers[i])
                        centers = new_centers
                
                return centers

        initial_centers = adaptive_recursive_cover(X, p)
        
        # Perform radius refinement
        for _ in range(20):
            distances = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
            nearest_center = np.argmin(distances, axis=1)
            new_centers = np.copy(initial_centers)
            for i in range(initial_centers.shape[0]):
                points_in_cluster = X[nearest_center == i]
                if points_in_cluster.size == 0:
                    new_centers[i] = np.mean(X, axis=0)
                else:
                    new_centers[i] = np.mean(points_in_cluster, axis=0) + 0.2 * (np.mean(points_in_cluster, axis=0) - initial_centers[i])
            initial_centers = new_centers
        
        # Perform additional refinement to improve center diversification
        for _ in range(15):
            distances = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
            nearest_center = np.argmin(distances, axis=1)
            new_centers = np.copy(initial_centers)
            for i in range(initial_centers.shape[0]):
                points_in_cluster = X[nearest_center == i]
                if points_in_cluster.size == 0:
                    new_centers[i] = np.mean(X, axis=0)
                else:
                    new_centers[i] = np.mean(points_in_cluster, axis=0) + 0.1 * (np.mean(points_in_cluster, axis=0) - initial_centers[i])
            initial_centers = new_centers
        
        # Final refinement with larger step size
        for _ in range(10):
            distances = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
            nearest_center = np.argmin(distances, axis=1)
            new_centers = np.copy(initial_centers)
            for i in range(initial_centers.shape[0]):
                points_in_cluster = X[nearest_center == i]
                if points_in_cluster.size == 0:
                    new_centers[i] = np.mean(X, axis=0)
                else:
                    new_centers[i] = np.mean(points_in_cluster, axis=0) + 0.5 * (np.mean(points_in_cluster, axis=0) - initial_centers[i])
            initial_centers = new_centers
        
        return initial_centers