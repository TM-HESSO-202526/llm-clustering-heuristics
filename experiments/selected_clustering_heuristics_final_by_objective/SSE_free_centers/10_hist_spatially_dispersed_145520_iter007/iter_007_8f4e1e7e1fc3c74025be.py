import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Main mechanism: Spatially Dispersed Clustering (SDC)
        # This algorithm uses a novel approach to disperse the centroids in the data space while maintaining a balance between the number of points assigned to each centroid and their spatial distribution.
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the set of centroids with a random selection of points
        centroids = X[rng.choice(len(X), size=p, replace=False)]
        
        # Repeat until convergence or a maximum number of iterations
        for _ in range(100):
            # Assign each point to the closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            min_indices = np.argmin(distances, axis=1)
            
            # Update the centroids to be the centroid of the points assigned to it
            new_centroids = np.copy(centroids)
            for i in range(p):
                points = X[min_indices == i]
                if len(points) > 0:
                    new_centroids[i] = np.mean(points, axis=0)
            
            # Disperse the centroids to maintain a balance between the number of points assigned to each centroid and their spatial distribution
            for i in range(p):
                points = X[min_indices == i]
                if len(points) > 0:
                    centroid = new_centroids[i]
                    # Calculate the dispersion factor based on the number of points assigned to the centroid and their spatial distribution
                    dispersion_factor = np.std(np.linalg.norm(points - centroid, axis=1)) / np.mean(np.linalg.norm(points - centroid, axis=1))
                    # Update the centroid to maintain a balance between the number of points assigned to it and their spatial distribution
                    new_centroids[i] = centroid + (rng.random(size=X.shape[1]) - 0.5) * dispersion_factor * np.std(points, axis=0)
            
            # Check for convergence
            if np.all(new_centroids == centroids):
                break
            
            # Update the centroids
            centroids = new_centroids
        
        # Return the centroids
        return centroids