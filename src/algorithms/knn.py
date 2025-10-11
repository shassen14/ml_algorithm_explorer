# src/algorithms/knn.py
import numpy as np
from collections import Counter


class SimpleKNNClassifier:
    """
    A from-scratch implementation of the K-Nearest Neighbors algorithm.
    This class is designed for educational purposes to demonstrate the core logic.
    """

    def __init__(self, k=3):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """'Fits' the model by memorizing the entire training dataset."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        """Helper function to calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """Predicts the class labels for a set of new data points."""
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x_test_point):
        """Predicts the label for a single new data point by finding its neighbors."""
        # 1. Calculate the distance from our new point to ALL training points.
        distances = [
            self._euclidean_distance(x_test_point, x_train_point)
            for x_train_point in self.X_train
        ]

        # 2. Get the indices of the 'k' training points with the smallest distances.
        k_nearest_indices = np.argsort(distances)[: self.k]

        # 3. Get the labels of these 'k' closest neighbors.
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # 4. Determine the most common label among the neighbors (majority vote).
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
