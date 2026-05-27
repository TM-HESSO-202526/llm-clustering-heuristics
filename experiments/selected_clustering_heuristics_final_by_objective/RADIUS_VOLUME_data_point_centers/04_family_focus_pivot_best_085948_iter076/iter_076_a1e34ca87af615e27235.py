import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        n, d = X.shape
        centers = []
        
        # Select initial medoid
        initial_medoid = X[rng.integers(n)]
        centers.append(initial_medoid)
        
        # Select remaining medoids
        for _ in range(p - 1):
            distances = np.full(n, np.inf)
            for center in centers:
                dist = np.linalg.norm(X - center, axis=1)
                distances = np.minimum(distances, dist)
            
            next_medoid_idx = np.argmax(distances)
            next_medoid = X[next_medoid_idx]
            centers.append(next_medoid)
        
        # Radius-covering medoid pivoting control
        for _ in range(150):  # Increased refinement iterations
            distances = np.full(n, np.inf)
            labels = np.zeros(n, dtype=int)
            for i, center in enumerate(centers):
                dist = np.linalg.norm(X - center, axis=1)
                mask = dist < distances
                distances[mask] = dist[mask]
                labels[mask] = i
            
            radii = np.zeros(p)
            for i in range(p):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    radii[i] = np.max(np.linalg.norm(cluster_points - centers[i], axis=1))
            
            largest_radii_idx = np.argsort(radii)[::-1][:min(7, p)]  # Focus on top clusters
            
            for idx in largest_radii_idx:
                cluster_points = X[labels == idx]
                if len(cluster_points) > 0:
                    min_max_distance = np.inf
                    new_medoid_idx = None
                    for i, point in enumerate(cluster_points):
                        max_distance = np.max(np.linalg.norm(cluster_points - point, axis=1))
                        if max_distance < min_max_distance:
                            min_max_distance = max_distance
                            new_medoid_idx = i
                    
                    if new_medoid_idx is not None:
                        new_medoid = cluster_points[new_medoid_idx]
                        centers[idx] = new_medoid
        
        # Ensure all centers are active
        distances = np.full(n, np.inf)
        labels = np.zeros(n, dtype=int)
        for i, center in enumerate(centers):
            dist = np.linalg.norm(X - center, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            labels[mask] = i
        
        # Remove empty centers and replace them with new ones
        active_centers = np.unique(labels)
        if len(active_centers) < p:
            new_centers = []
            for i in range(p):
                if i in active_centers:
                    new_centers.append(centers[i])
                else:
                    farthest_points = X[np.argsort(distances)[::-1]]
                    new_center = farthest_points[0]
                    new_centers.append(new_center)
            centers = new_centers
        
        # Additional step to reduce largest radii
        for _ in range(30):
            distances = np.full(n, np.inf)
            labels = np.zeros(n, dtype=int)
            for i, center in enumerate(centers):
                dist = np.linalg.norm(X - center, axis=1)
                mask = dist < distances
                distances[mask] = dist[mask]
                labels[mask] = i
            
            radii = np.zeros(p)
            for i in range(p):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    radii[i] = np.max(np.linalg.norm(cluster_points - centers[i], axis=1))
            
            largest_radii_idx = np.argsort(radii)[::-1][:1]  # Focus on the cluster with the largest radius
            
            for idx in largest_radii_idx:
                cluster_points = X[labels == idx]
                if len(cluster_points) > 0:
                    min_max_distance = np.inf
                    new_medoid_idx = None
                    for i, point in enumerate(cluster_points):
                        max_distance = np.max(np.linalg.norm(cluster_points - point, axis=1))
                        if max_distance < min_max_distance:
                            min_max_distance = max_distance
                            new_medoid_idx = i
                    
                    if new_medoid_idx is not None:
                        new_medoid = cluster_points[new_medoid_idx]
                        centers[idx] = new_medoid
        
        return np.array(centers)