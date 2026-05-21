import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        n, d = X.shape
        centers = np.zeros((p, d))
        
        # Initialize the first center as a random point with the maximum distance to the origin
        distances_to_origin = np.linalg.norm(X, axis=1)
        idx = np.argmax(distances_to_origin)
        centers[0] = X[idx]
        
        # Iteratively select the next center
        for i in range(1, p):
            # Compute the distance from each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers[:i], axis=2)
            min_distances = np.min(distances, axis=1)
            
            # Compute the score for each point
            scores = np.maximum(min_distances, np.max(min_distances) / 2)
            
            # Select the point with the maximum score as the next center
            idx = np.argmax(scores)
            centers[i] = X[idx]
        
        # Perform a local search to improve the centers
        for _ in range(10):
            new_centers = np.copy(centers)
            for i in range(p):
                # Compute the distance from each point to the current center
                distances = np.linalg.norm(X - centers[i], axis=1)
                idx = np.argmin(distances)
                new_centers[i] = X[idx]
            # Check if the new centers are an improvement
            if np.all(new_centers == centers):
                break
            centers = new_centers
        
        # Perform high-radius repair
        for _ in range(5):
            new_centers = np.copy(centers)
            for i in range(p):
                # Compute the distance from each point to the current center
                distances = np.linalg.norm(X - centers[i], axis=1)
                idx = np.argmax(distances)
                max_distance = np.max(distances)
                if max_distance > np.mean(distances) * 2:
                    new_centers[i] = X[idx]
            # Check if the new centers are an improvement
            if np.all(new_centers == centers):
                break
            centers = new_centers
        
        # Additional improvement: recompute the distance from each point to the nearest center
        # and reassign the points to the centers to ensure the clusters are well-separated
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        min_distances = np.min(distances, axis=1)
        min_indices = np.argmin(distances, axis=1)
        
        # Recompute the centers as the point with the minimum distance to the centroid of each cluster
        for i in range(p):
            cluster_points = X[min_indices == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
                idx = np.argmin(distances_to_centroid)
                centers[i] = cluster_points[idx]
        
        return centers