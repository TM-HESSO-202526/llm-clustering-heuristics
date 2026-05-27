import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        n, d = X.shape
        centers = np.zeros((p, d))
        
        # Initialize the first center randomly
        centers[0] = X[rng.choice(n)]
        
        min_dist = np.linalg.norm(X - centers[0], axis=1)
        
        for i in range(1, p):
            # Calculate the farthest point from the existing centers
            farthest_point_idx = np.argmax(min_dist)
            farthest_point = X[farthest_point_idx]
            
            # Calculate the coverage score for each point
            coverage_scores = np.zeros(n)
            for j in range(n):
                point = X[j]
                dist_to_farthest = np.linalg.norm(point - farthest_point)
                dist_to_closest_center = np.min(np.linalg.norm(point - centers[:i], axis=1))
                if dist_to_farthest == 0:
                    coverage_scores[j] = 0
                else:
                    coverage_scores[j] = dist_to_farthest * (1 - dist_to_closest_center / (dist_to_farthest + 1e-6)) * (1 - np.sum(np.linalg.norm(centers[:i] - point, axis=1)) / (i + 1e-6))
            
            # Select the point with the highest coverage score as the new center
            new_center_idx = np.argmax(coverage_scores)
            centers[i] = X[new_center_idx]
            
            # Update the minimum distances
            dist_to_new_center = np.linalg.norm(X - centers[i], axis=1)
            min_dist = np.minimum(min_dist, dist_to_new_center)
        
        # Apply bounded selected-point repair
        for i in range(p):
            # Calculate the distance from each point to the current center
            dist_to_center = np.linalg.norm(X - centers[i], axis=1)
            
            # Find the point with the minimum distance to the current center
            closest_point_idx = np.argmin(dist_to_center)
            closest_point = X[closest_point_idx]
            
            # Update the center if the closest point is not the current center
            if not np.array_equal(closest_point, centers[i]):
                centers[i] = closest_point
        
        # Additional bounded refinement
        for _ in range(2):  # Perform 2 iterations of refinement
            for i in range(p):
                # Calculate the distance from each point to the current center
                dist_to_center = np.linalg.norm(X - centers[i], axis=1)
                
                # Find the point with the minimum distance to the current center
                closest_point_idx = np.argmin(dist_to_center)
                closest_point = X[closest_point_idx]
                
                # Update the center if the closest point is not the current center
                if not np.array_equal(closest_point, centers[i]):
                    centers[i] = closest_point
        
        return centers