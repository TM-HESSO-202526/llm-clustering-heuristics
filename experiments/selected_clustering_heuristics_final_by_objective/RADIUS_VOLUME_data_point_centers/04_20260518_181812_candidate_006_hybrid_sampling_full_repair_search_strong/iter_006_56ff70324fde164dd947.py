import numpy as np

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        n, d = X.shape
        if rng is None:
            rng = np.random.default_rng()

        # Sample construction
        sample_size = min(n, 10 * p)
        indices = rng.choice(n, size=sample_size, replace=False)
        S = X[indices]

        # Initial medoids using k-means++
        medoids = self.kmeans_plus_plus(S, p, rng)

        # Evaluate assignments on the full X
        assignments = self.assign_to_nearest(X, medoids)
        radii = self.calculate_radii(X, assignments, medoids)

        # Repair high-radius clusters
        for _ in range(10):  # Increased number of repair iterations
            high_radius_indices = np.argsort(radii)[::-1][:max(1, p // 3)]  # Increased number of high-radius clusters to repair
            for i in high_radius_indices:
                cluster_points = X[assignments == i]
                new_medoid = self.find_best_medoid(cluster_points, medoids[i], cluster_points)
                medoids[i] = new_medoid

            assignments = self.assign_to_nearest(X, medoids)
            radii = self.calculate_radii(X, assignments, medoids)

        # Additional refinement
        for _ in range(3):
            for i in range(len(medoids)):
                cluster_points = X[assignments == i]
                best_medoid = medoids[i]
                best_radius = np.max([self.calculate_distance(x, medoids[i]) for x in cluster_points])
                for point in cluster_points:
                    radius = np.max([self.calculate_distance(x, point) for x in cluster_points])
                    if radius < best_radius:
                        best_medoid = point
                        best_radius = radius
                medoids[i] = best_medoid

            assignments = self.assign_to_nearest(X, medoids)
            radii = self.calculate_radii(X, assignments, medoids)

        return medoids

    def kmeans_plus_plus(self, X, k, rng):
        medoids = [X[rng.choice(X.shape[0])]]
        for _ in range(k - 1):
            distances = np.array([self.calculate_distance(x, m) for x in X for m in medoids]).reshape(X.shape[0], -1)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            index = rng.choice(X.shape[0], p=probabilities)
            medoids.append(X[index])
        return np.array(medoids)

    def assign_to_nearest(self, X, medoids):
        distances = np.array([self.calculate_distance(x, m) for x in X for m in medoids]).reshape(X.shape[0], -1)
        return np.argmin(distances, axis=1)

    def calculate_radii(self, X, assignments, medoids):
        radii = []
        for i in range(len(medoids)):
            cluster_points = X[assignments == i]
            if cluster_points.size == 0:
                radii.append(0)
            else:
                distances = np.array([self.calculate_distance(x, medoids[i]) for x in cluster_points])
                radii.append(np.max(distances))
        return np.array(radii)

    def find_best_medoid(self, cluster_points, old_medoid, all_points):
        best_medoid = old_medoid
        best_radius = np.max([self.calculate_distance(x, old_medoid) for x in cluster_points])
        for point in cluster_points:
            radius = np.max([self.calculate_distance(x, point) for x in cluster_points])
            if radius < best_radius:
                best_medoid = point
                best_radius = radius
        return best_medoid

    def calculate_distance(self, x, y):
        return np.linalg.norm(x - y)