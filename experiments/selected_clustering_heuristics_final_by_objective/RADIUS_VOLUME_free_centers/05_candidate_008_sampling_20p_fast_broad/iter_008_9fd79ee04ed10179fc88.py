import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the first center as the point with the maximum distance to the mean of the sample
        mean = np.mean(X, axis=0)
        distances_to_mean = np.linalg.norm(X - mean, axis=1)
        first_center_index = np.argmax(distances_to_mean)
        first_center = X[first_center_index]
        centers = [first_center]
        
        # Loop to find the remaining centers
        for _ in range(p - 1):
            # Calculate the squared distances from each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2)
            min_distances = np.min(distances, axis=1)
            
            # Calculate the probability of selecting each point as the next center
            probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
            
            # Introduce enhanced exploration by choosing between the highest probability and a random point
            index_max_prob = np.argmax(probabilities)
            if rng.random() < 0.6:
                index = index_max_prob
            else:
                index = rng.choice(X.shape[0], p=probabilities)
            
            new_center = X[index]
            
            # Add the new center to the list of centers
            centers.append(new_center)
        
        # Stack the list of centers into a numpy array
        centers = np.array(centers)
        
        # Perform a local optimization to refine the centers
        for _ in range(8):
            # Assign each point to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update the centers as the mean of their assigned points with a random perturbation
            for i in range(p):
                points_assigned_to_i = X[assignments == i]
                if points_assigned_to_i.size > 0:
                    centers[i] = np.mean(points_assigned_to_i, axis=0) + rng.normal(0, 0.03, size=X.shape[1])
        
        # Additional refinement step with reduced perturbation
        for _ in range(4):
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)
            for i in range(p):
                points_assigned_to_i = X[assignments == i]
                if points_assigned_to_i.size > 0:
                    centers[i] = np.mean(points_assigned_to_i, axis=0) + rng.normal(0, 0.005, size=X.shape[1])
        
        # Perform cluster size balancing
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        assignments = np.argmin(distances, axis=1)
        cluster_sizes = np.bincount(assignments, minlength=p)
        max_cluster_size = np.max(cluster_sizes)
        
        # Adjust centers based on cluster size imbalance
        for i in range(p):
            if cluster_sizes[i] > max_cluster_size * 1.2:
                points_assigned_to_i = X[assignments == i]
                new_center = np.mean(points_assigned_to_i, axis=0) + rng.normal(0, 0.01, size=X.shape[1])
                centers[i] = new_center
        
        return centers