import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the first center as a random point
        centers = X[rng.integers(X.shape[0]), np.newaxis, :]
        
        for _ in range(1, p):
            # Calculate the squared distances from each point to the nearest center
            dist_sq = np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2)
            dist_sq = np.min(dist_sq, axis=1)
            
            # Select the next center based on the maximum distance
            next_center_idx = np.argmax(dist_sq)
            
            # Add the next center to the list of centers
            centers = np.vstack((centers, X[next_center_idx, np.newaxis, :]))
        
        # Local search to improve the centers
        for _ in range(5):
            for i in range(centers.shape[0]):
                # Calculate the squared distances from each point to the current centers
                dist_sq = np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2)
                dist_sq = np.min(dist_sq, axis=1)
                
                # Calculate the squared distances from each point to the current centers excluding the i-th center
                dist_sq_excluding_i = np.sum((X[:, np.newaxis, :] - np.delete(centers, i, axis=0)) ** 2, axis=2)
                if dist_sq_excluding_i.size == 0:
                    dist_sq_excluding_i = np.inf * np.ones(X.shape[0])
                else:
                    dist_sq_excluding_i = np.min(dist_sq_excluding_i, axis=1)
                
                # Calculate the gain of replacing the i-th center with each point
                gain = dist_sq - dist_sq_excluding_i
                
                # Select the point with the maximum gain
                max_gain_idx = np.argmax(gain)
                
                # If the gain is positive, replace the i-th center with the selected point
                if gain[max_gain_idx] > 0:
                    centers[i] = X[max_gain_idx, np.newaxis, :]
        
        return centers.squeeze()