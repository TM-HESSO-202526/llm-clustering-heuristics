import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize centers randomly
        centers = X[rng.integers(0, X.shape[0], size=p)]
        
        for _ in range(100):  # number of refinement steps
            # Assign points to nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Calculate cluster radii
            radii = np.array([np.max(distances[assignments == j, j]) if np.sum(assignments == j) > 0 else 0 for j in range(centers.shape[0])])
            
            # Split largest-radius cluster
            if radii.size < p:
                largest_radius_idx = np.argmax(radii)
                if radii[largest_radius_idx] > 0:
                    farthest_points = X[assignments == largest_radius_idx]
                    farthest_points_distances = np.linalg.norm(farthest_points - centers[largest_radius_idx], axis=1)
                    farthest_point_idx = np.argmax(farthest_points_distances)
                    new_center = farthest_points[farthest_point_idx]
                    centers = np.vstack((centers, new_center))
            
            # Repair empty centers
            for j in range(centers.shape[0]):
                if np.sum(assignments == j) == 0:
                    farthest_point_idx = np.argmax(np.min(distances, axis=1))
                    centers[j] = X[farthest_point_idx]
            
            # Reduce large cluster radii
            for j in range(centers.shape[0]):
                if radii[j] > np.mean(radii) and radii[j] > 0:
                    points_in_cluster = X[assignments == j]
                    if points_in_cluster.size > 0:
                        centroid = np.mean(points_in_cluster, axis=0)
                        centers[j] = centroid
            
            # Remove extra centers if needed
            if centers.shape[0] > p:
                radii = np.array([np.max(distances[assignments == j, j]) if np.sum(assignments == j) > 0 else 0 for j in range(centers.shape[0])])
                smallest_radius_idx = np.argmin(radii)
                centers = np.delete(centers, smallest_radius_idx, axis=0)
        
        # Assign points to nearest center one last time
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        assignments = np.argmin(distances, axis=1)
        
        # Repair empty centers one last time
        for j in range(centers.shape[0]):
            if np.sum(assignments == j) == 0:
                farthest_point_idx = np.argmax(np.min(distances, axis=1))
                centers[j] = X[farthest_point_idx]
        
        # Add additional refinement steps to ensure convergence
        for _ in range(50):
            # Assign points to nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Calculate cluster radii
            radii = np.array([np.max(distances[assignments == j, j]) if np.sum(assignments == j) > 0 else 0 for j in range(centers.shape[0])])
            
            # Reduce large cluster radii
            for j in range(centers.shape[0]):
                if radii[j] > np.mean(radii) and radii[j] > 0:
                    points_in_cluster = X[assignments == j]
                    if points_in_cluster.size > 0:
                        centroid = np.mean(points_in_cluster, axis=0)
                        centers[j] = centroid
        
        # Final adjustment to ensure p centers
        if centers.shape[0] < p:
            for _ in range(p - centers.shape[0]):
                farthest_point_idx = np.argmax(np.min(distances, axis=1))
                new_center = X[farthest_point_idx]
                centers = np.vstack((centers, new_center))
        
        # One last refinement step
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        assignments = np.argmin(distances, axis=1)
        for j in range(centers.shape[0]):
            if np.sum(assignments == j) > 0:
                points_in_cluster = X[assignments == j]
                centroid = np.mean(points_in_cluster, axis=0)
                centers[j] = centroid
        
        return centers