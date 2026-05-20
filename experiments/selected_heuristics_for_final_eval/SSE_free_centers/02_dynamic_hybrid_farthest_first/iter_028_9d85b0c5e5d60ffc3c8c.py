import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        n, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Define the scoring function
        def score(X, centers):
            return np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1) ** 2)

        # Define the Lloyd Refinement with adaptive stopping
        def lloyd_refine(X, centers, max_iterations=100):
            prev_centers = centers.copy()
            for _ in range(max_iterations):  
                # Assign points to nearest center
                labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)

                # Update centers
                new_centers = np.array([X[labels == i].mean(axis=0) for i in range(p)])

                # Check for convergence
                if np.all(np.linalg.norm(prev_centers - new_centers, axis=1) < 1e-6):
                    break
                prev_centers = centers
                centers = new_centers

            return centers

        # Define the hybrid refinement
        def hybrid_refine(X, centers):
            # Start with standard Lloyd refinement
            centers = lloyd_refine(X, centers, max_iterations=10)
            # Then use a more aggressive refinement with fewer iterations but smaller convergence threshold
            centers = lloyd_refine(X, centers, max_iterations=5)
            return centers

        # Define the enhanced dynamic hybrid refinement
        def enhanced_hybrid_refine(X, centers, p):
            if p <= 20:
                max_iterations = 35
            elif p <= 50:
                max_iterations = 30
            else:
                max_iterations = 25
            centers = hybrid_refine(X, centers)
            centers = lloyd_refine(X, centers, max_iterations=max_iterations)  # Additional refinement step
            return centers

        # Farthest-First Initialization with randomized restart and enhanced scoring
        best_centers = None
        best_score = np.inf
        for _ in range(40):  # Increased restarts for better initialization
            centers = np.zeros((p, d))
            centers[0] = X[rng.choice(n)]
            min_d2 = np.linalg.norm(X - centers[0], axis=1) ** 2
            for i in range(1, p):
                centers[i] = X[np.argmax(min_d2)]
                new_d2 = np.linalg.norm(X - centers[i], axis=1) ** 2
                min_d2 = np.minimum(min_d2, new_d2)

            # Enhanced dynamic hybrid refinement
            centers = enhanced_hybrid_refine(X, centers, p)

            # Evaluate the score
            current_score = score(X, centers)
            if current_score < best_score:
                best_score = current_score
                best_centers = centers

            # Adaptive restart strategy based on the current score and number of restarts
            if current_score < best_score * 0.95 or _ > 30:
                break

        # Additional adaptive scoring refinement
        if p > 20:
            for _ in range(20):  # Increased refinement iterations
                labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - best_centers, axis=2), axis=1)
                new_centers = np.array([X[labels == i].mean(axis=0) for i in range(p)])
                new_score = score(X, new_centers)
                if new_score < best_score:
                    best_score = new_score
                    best_centers = new_centers
                else:
                    break

        # Additional iterative refinement
        for _ in range(5):  # Additional refinement iterations
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - best_centers, axis=2), axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(p)])
            new_score = score(X, new_centers)
            if new_score < best_score:
                best_score = new_score
                best_centers = new_centers

        return best_centers