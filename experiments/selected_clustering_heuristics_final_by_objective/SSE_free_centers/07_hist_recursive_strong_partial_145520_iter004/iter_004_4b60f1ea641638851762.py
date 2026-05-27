import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        # Main mechanism: Hierarchical partition-based clustering
        # This algorithm uses a hierarchical approach to divide the data into partitions and then find the optimal centers.
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the set of centers
        centers = []
        
        # Repeat until we have p centers
        for _ in range(p):
            # If there are no centers, choose a random point
            if not centers:
                center = X[rng.integers(len(X))]
            else:
                # Find the point that is farthest from the existing centers
                distances = np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2)
                min_distances = np.min(distances, axis=1)
                center_index = np.argmax(min_distances)
                center = X[center_index]
            
            # Add the center to the set of centers
            centers.append(center)
            
            # Divide the data space into partitions
            partitions = self.divide_partitions(X, centers)
            
            # Update the center to be the mean of the points in the partition
            for i in range(len(centers)):
                partition = partitions[i]
                if len(partition) > 0:
                    centers[i] = np.mean(partition, axis=0)
        
        # Return the centers
        return np.array(centers)

    def divide_partitions(self, X, centers):
        # Divide the data space into partitions based on the centers
        partitions = [[] for _ in range(len(centers))]
        for point in X:
            min_distance = np.inf
            min_index = -1
            for i, center in enumerate(centers):
                distance = np.linalg.norm(point - center)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            partitions[min_index].append(point)
        return partitions