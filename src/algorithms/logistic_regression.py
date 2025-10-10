# src/algorithms/logistic_regression.py
import numpy as np


class SimpleLogisticRegression:
    """
    A from-scratch implementation of Logistic Regression using Gradient Descent.
    This is meant for binary decisions and not multiclass decisions
    Designed for educational purposes.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """The S-shaped activation function that maps any value to a probability between 0 and 1."""
        # Clip z to avoid overflow in np.exp for very large negative numbers
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Learns the best weights and bias from the data using Gradient Descent."""
        n_samples, n_features = X.shape

        # 1. Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent loop
        for _ in range(self.n_iterations):
            # Calculate the linear model (z = w*x + b)
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply the sigmoid function to get predictions (probabilities)
            y_predicted = self._sigmoid(linear_model)

            # Calculate the gradients (the direction of steepest ascent of the error)
            dw = (1 / n_samples) * np.dot(
                X.T, (y_predicted - y)
            )  # Derivative for weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Derivative for bias

            # Update the parameters by taking a small step in the opposite direction of the gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_probabilities(self, X):
        """Returns the probability for the positive class for each sample."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Makes a final class prediction based on a probability threshold."""
        probabilities = self.predict_probabilities(X)
        return [1 if i > threshold else 0 for i in probabilities]
