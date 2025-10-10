# ui/components/linear_model_explainer.py
from sklearn.metrics import log_loss
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.algorithms.logistic_regression import SimpleLogisticRegression
import inspect


def render(df, target_column):
    """Renders the full, improved learning experience for Logistic Regression."""

    # --- LEVEL 1: The Story & The Goal ---
    st.markdown("### The Goal: Find the Best Fit to Minimize Error")
    st.info(
        """
    Machine learning is an optimization game. The goal of "fitting" is to find the parameters for the **S-shaped 'sigmoid' curve** that make the model's predictions as close to the real data as possible. We measure this "closeness" with a score called **Log Loss**.

    **Your challenge:** Can you adjust the sliders to find the curve with the **lowest possible Log Loss score**?
    """
    )
    st.markdown("---")

    # --- LEVEL 2: The Visual & Code-Level Intuition ---

    # New Top-to-Bottom Layout
    render_interactive_demo(df, target_column)
    st.markdown("---")
    render_code_explanation()

    st.markdown("---")
    with st.expander("Future Enhancements: Visualizing in Higher Dimensions"):
        st.info(
            """
        This interactive plot simplifies the problem to a single feature to build intuition. 
        However, the real power of a model like Logistic Regression comes from its ability to combine **many features** at once.

        **Future Work:**
        - **2D Interactive Plot:** A next version could use PCA to project the data into 2D and allow the user to interactively 
        draw a linear decision boundary (a straight line) to separate the classes.
        - **3D Interactive Plot:** An even more advanced version could create a 3D scatter plot (using the first three principal 
        components) and visualize the decision boundary as a 2D plane slicing through the 3D space.

        These higher-dimensional visualizations help build intuition for how linear models operate in a multidimensional feature space.
        """
        )


def render_interactive_demo(df, target_column):
    st.subheader("Interactive Playground")

    # --- SETUP: Define the Binary Problem ---
    # We must explicitly define which two classes we are comparing.
    all_classes = df[target_column].unique()

    col1, col2 = st.columns(2)
    with col1:
        # Let the user choose the "negative" class (will be plotted at y=0)
        neg_class = st.selectbox(
            "Select the Negative Class (plotted at y=0):", all_classes, index=0
        )
    with col2:
        # Let the user choose the "positive" class (will be plotted at y=1)
        pos_class = st.selectbox(
            "Select the Positive Class (plotted at y=1):",
            all_classes,
            index=min(1, len(all_classes) - 1),
        )

    feature = st.selectbox(
        "Select a numerical feature to explore:",
        df.select_dtypes(include=np.number).columns,
    )

    if feature and neg_class != pos_class:
        # Filter the DataFrame to only include the two selected classes
        binary_df = df[df[target_column].isin([neg_class, pos_class])]
        X = binary_df[[feature]].dropna()
        y = binary_df.loc[X.index, target_column]
        y_encoded = (y == pos_class).astype(int)

        # Train a model on this specific binary problem to get the TRUE optimal fit
        simple_model = LogisticRegression().fit(X, y_encoded)
        optimal_coef = simple_model.coef_[0][0]
        optimal_intercept = simple_model.intercept_[0]

        x_mean = float(X.mean().item())
        x_min = float(X.min().item())
        x_max = float(X.max().item())

        # Sliders for user interaction
        slope = st.slider("1. Adjust Slope (Steepness)", -2.0, 2.0, 0.1, 0.01)
        boundary_pos = st.slider(
            "2. Adjust Decision Boundary (Position)", x_min, x_max, x_mean
        )

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate points for the sigmoid curve
        x_range = np.linspace(X.min().item(), X.max().item(), 300)

        # The formula for the sigmoid curve's input (z) is z = slope * (x - intercept)
        # This makes the "intercept" slider intuitively move the curve left and right.
        z = slope * (x_range - boundary_pos)
        y_sigmoid = 1 / (1 + np.exp(-z))

        # Calculate your model's predictions for the actual data points
        your_preds_proba = 1 / (1 + np.exp(-(slope * (X[feature] - boundary_pos))))
        # Add a small epsilon to prevent log(0) errors
        epsilon = 1e-15
        your_preds_proba = np.clip(your_preds_proba, epsilon, 1 - epsilon)
        current_log_loss = log_loss(y_encoded, your_preds_proba)

        # Calculate optimal log loss
        optimal_preds_proba = simple_model.predict_proba(X)[:, 1]
        optimal_log_loss = log_loss(y_encoded, optimal_preds_proba)

        # --- DISPLAY THE "GAME" SCORES ---
        st.markdown("#### Your Score (Lower is Better!)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Your Log Loss",
                value=f"{current_log_loss:.4f}",
                delta=f"{current_log_loss - optimal_log_loss:.4f} vs Optimal",
            )
        with col2:
            st.metric(label="Optimal Log Loss", value=f"{optimal_log_loss:.4f}")

        # Shade the background based on the decision boundary
        ax.axvspan(
            x_min,
            boundary_pos,
            facecolor="royalblue",
            alpha=0.15,
            label=f"Predicted '{neg_class}'",
        )
        ax.axvspan(
            boundary_pos,
            x_max,
            facecolor="tomato",
            alpha=0.15,
            label=f"Predicted '{pos_class}'",
        )

        # Plot data points
        ax.scatter(
            X[y_encoded == 0],
            y_encoded[y_encoded == 0],
            color="darkblue",
            label=neg_class,
            alpha=0.7,
        )
        ax.scatter(
            X[y_encoded == 1],
            y_encoded[y_encoded == 1],
            color="darkred",
            label=pos_class,
            alpha=0.7,
        )

        # Plot fits and boundaries
        ax.plot(x_range, y_sigmoid, color="black", lw=3, label="Your Fit")
        ax.axvline(
            x=boundary_pos,
            color="black",
            linestyle="--",
            label="Your Decision Boundary",
        )

        if st.checkbox("Show Optimal Fit", value=True):
            opt_boundary = (
                -optimal_intercept / optimal_coef if optimal_coef != 0 else x_mean
            )
            z_opt = optimal_coef * (x_range - opt_boundary)
            y_optimal = 1 / (1 + np.exp(-z_opt))
            ax.plot(
                x_range,
                y_optimal,
                color="green",
                lw=3,
                linestyle=":",
                label="Optimal Fit",
            )
            ax.axvline(
                x=opt_boundary, color="green", linestyle=":", label="Optimal Boundary"
            )

        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(
            [f"Certain '{neg_class}'", "Uncertain", f"Certain '{pos_class}'"]
        )
        ax.set_xlabel(feature)
        ax.set_ylabel(f"Probability of '{pos_class}'")
        ax.set_title(f"Interactive Logistic Regression Fit for '{feature}'")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        fig.tight_layout()
        st.pyplot(fig)


def render_code_explanation():
    st.subheader("The Code Behind It")
    st.info(
        "This from-scratch code uses **Gradient Descent** to 'learn' the best-fitting curve."
    )

    with st.expander("Show From-Scratch Python Code"):
        code_string = inspect.getsource(SimpleLogisticRegression)
        st.code(code_string, language="python")

    st.markdown("---")
    st.warning(
        """
    **A Note on Naming:** You'll notice our method is named `predict_probabilities`. 
    This is for maximum clarity. 
    
    The industry-standard library, **Scikit-learn**, calls this same method `predict_proba`. 
    We chose the more descriptive name to make the code's purpose immediately obvious.
    """,
        icon="💡",
    )

    st.markdown(
        """
    #### Key Concepts in the Code:
    - **`_sigmoid` function:** Creates the S-shaped probability curve.
    - **`fit` method:** The learning process. The `for` loop iteratively adjusts `weights` and `bias` 
    to minimize prediction error. This is **Gradient Descent**.
    - **`predict_probabilities` method:** Calculates the model's confidence score (probability) for each data point.
    - **`predict` method:** Takes the final probability and uses a 0.5 threshold to make a final "Yes" or "No" decision.
    """
    )

    st.markdown("---")
    with st.expander("How does this work for more than two classes?"):
        st.markdown(
            """
        Our from-scratch implementation and the interactive demo above are designed for **binary classification**. They find a single decision boundary to separate two classes.
        
        To handle **multiclass** problems, two main strategies are used:

        #### 1. One-vs-Rest (OvR)
        - The algorithm trains one binary classifier for each class (e.g., 'Class A' vs. 'Not A', 'Class B' vs. 'Not B', etc.).
        - To make a prediction, the new data point is shown to all classifiers, and the one that is most confident "wins".
        - **This is the default method used by Scikit-learn's `LogisticRegression`.**

        #### 2. Multinomial (Softmax Regression)
        - A more direct approach where the model learns one set of weights *per class*.
        - It uses the **Softmax function** (a generalization of the sigmoid) to output a probability distribution across all classes at once.
        - This is the standard for neural network classifiers. You can enable it in Scikit-learn by setting `multi_class='multinomial'`.
        """
        )
