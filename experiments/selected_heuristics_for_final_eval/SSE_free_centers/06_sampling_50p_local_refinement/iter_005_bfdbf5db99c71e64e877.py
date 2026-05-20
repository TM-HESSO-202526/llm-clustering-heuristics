import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        m, d = X.shape
        
        # Use a hybrid approach to initialize centers
        if p > m:
            # If p > m, return p centers by sampling/reusing points with replacement
            centers = X[rng.choice(m, size=p, replace=True)]
        else:
            # Otherwise, use k-means++ initialization with a twist
            centers = np.zeros((p, d))
            centers[0] = X[rng.choice(m)]
            
            for i in range(1, p):
                # Compute squared distances from each point to the closest center
                dist2 = np.full(m, np.inf)
                for j in range(i):
                    dist2 = np.minimum(dist2, np.sum((X - centers[j]) ** 2, axis=1))
                
                # Choose the next center with probability proportional to the squared distance
                # and a bonus for points that are far from existing centers
                bonus = np.maximum(0, 1 - (dist2 / np.max(dist2)))
                probs = (dist2 + bonus) / np.sum(dist2 + bonus)
                centers[i] = X[rng.choice(m, p=np.clip(probs, a_min=0, a_max=1))]
        
        # Perform a small bounded refinement with multiple iterations
        for _ in range(10):
            # Compute assignments to the closest center
            assign = np.argmin(np.sum((X[:, np.newaxis] - centers) ** 2, axis=2), axis=1)
            
            # Update centers as the mean of assigned points
            for i in range(p):
                assigned_points = X[assign == i]
                if len(assigned_points) > 0:
                    centers[i] = np.mean(assigned_points, axis=0)
        
        return centers