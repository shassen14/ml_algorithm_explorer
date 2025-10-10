# src/content/algorithm_descriptions.py

# This file stores the descriptive content for the "Algorithm Explainers" tab.
# This keeps the UI code clean and makes it easy to add or edit content.

ALGORITHM_INFO = {
    "Logistic Regression": {
        "pros": [
            "**Highly Interpretable:** The model's coefficients directly show the influence and direction of each feature's impact, making it easy to explain.",
            "**Fast and Efficient:** It's computationally inexpensive to train and predict, making it ideal for large datasets or low-latency applications.",
            "**Provides Probabilities:** Outputs well-calibrated probabilities, which are useful for ranking or when uncertainty is important.",
            "**Excellent Baseline:** Serves as a strong, simple baseline model to measure more complex models against.",
        ],
        "cons": [
            "**Assumes Linearity:** It can only learn a linear decision boundary. If the true relationship is complex and non-linear, it will perform poorly.",
            "**Sensitive to Multicollinearity:** Performance can suffer if features are highly correlated with each other.",
            "**May Be Outperformed:** Often less accurate than more complex, non-linear models like Random Forests or Gradient Boosting on complex datasets.",
        ],
    },
    "K-Nearest Neighbors": {
        "pros": [
            "**Simple and Intuitive:** The 'voting' analogy is very easy to understand and explain.",
            "**Naturally Non-Linear:** Can learn complex, irregular decision boundaries without any special configuration.",
            "**No 'Training' Phase:** The `fit` step is instant because it just stores the data, making it adaptable to new data on the fly.",
        ],
        "cons": [
            "**Computationally Expensive at Prediction:** Must calculate distances to all training points for every new prediction, making it very slow for large datasets.",
            "**The 'Curse of Dimensionality':** Performance degrades significantly as the number of features increases because the concept of 'distance' becomes less meaningful.",
            "**Sensitive to Feature Scaling:** Features with large scales (like 'salary') will dominate features with small scales (like 'age') unless the data is normalized.",
            "**Sensitive to Irrelevant Features:** Useless features can mislead the distance calculation, hurting performance.",
        ],
    },
    "Random Forest": {
        "pros": [
            "**Very Powerful and Accurate:** Often provides high accuracy right out of the box.",
            "**Robust to Overfitting:** By averaging many trees, it's much less likely to overfit than a single decision tree.",
            "**Handles Non-Linearity and Interactions:** Naturally captures complex relationships between features.",
            "**Resistant to Outliers and Irrelevant Features:** The random sampling of data and features makes it robust to noisy data.",
        ],
        "cons": [
            "**Less Interpretable (Black Box):** It's difficult to understand the exact reasoning behind a prediction from hundreds of trees.",
            "**Can be Slow to Train:** Training many deep trees on a large dataset can be computationally intensive.",
            "**May Not Extrapolate Well:** As a tree-based model, it cannot predict values outside the range seen in the training data.",
        ],
    },
    # Add entries for XGBoost, etc. here
}
