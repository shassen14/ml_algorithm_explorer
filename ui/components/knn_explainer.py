# ui/components/knn_explainer.py
import streamlit as st
from src.algorithms.knn import SimpleKNNClassifier
from src.evaluation.specific_plots.classification import plot_knn_neighbors
import inspect

from src.schemas import DisplayContext


def render(context: DisplayContext):
    """
    Renders the full learning experience for k-NN using explicitly passed arguments.
    """
    result = context.result
    processed_data = context.processed_data

    # --- LEVEL 1: The Story & The Goal ---
    st.markdown("### The Story: Judging by the Company You Keep")
    st.info(
        """
    Imagine you're trying to guess if a person likes action movies. You don't know anything about them, but you know about their **5 closest friends**. If 4 of those 5 friends love action movies, you'd probably guess that they do too.
    
    k-NN works exactly like this. Its goal is to classify a new, unknown data point based on a **majority vote of its 'k' nearest neighbors** from the data it has already seen.
    """
    )
    st.markdown("---")

    # --- LEVEL 2: The Visual & Code-Level Intuition ---
    tab1, tab2 = st.tabs(["Interactive Demo", "The Code Behind It"])

    with tab1:
        st.subheader("The k-NN Neighbor Inspector")
        st.write(
            "This plot visualizes the k-NN decision process for a single point from your test set. It is shown in a 2D space created by PCA for easy viewing of numerical features."
        )

        k_value = result.pipeline.named_steps["classifier"].n_neighbors
        point_index = st.slider(
            "Select a test point to inspect:",
            0,
            len(processed_data["X_test"]) - 1,
            0,
            key="knn_inspector_slider",
        )

        with st.spinner("Generating neighbor plot..."):
            fig = plot_knn_neighbors(
                processed_data["X_train"],
                result.y_train_encoded,
                processed_data["X_test"],
                point_index,
                k_value,
                result.class_names,
            )
            if fig:
                st.pyplot(fig)
            else:
                st.error(
                    "Could not generate the neighbor plot. This can happen if the data has no numerical features for PCA."
                )

    with tab2:
        st.markdown(
            """
        > "k-Nearest Neighbors is a simple, non-parametric **'lazy learning' algorithm** used for both 
        classification and regression.
        >
        > For classification, the process is straightforward:
        > 1.  **Memorization:** During the 'training' phase, the model simply stores the entire training 
        dataset. There's no actual learning of a model function.
        > 2.  **Prediction:** To classify a new, unseen data point, it calculates the **distance** 
        (commonly Euclidean distance) from this new point to every single point in the stored training data.
        > 3.  **Voting:** It identifies the 'k' closest points (the 'nearest neighbors') and takes a 
        **majority vote** among their labels. The most common label among the neighbors is assigned as 
        the prediction for the new point.
        >
        > It's an intuitive, instance-based approach, where the prediction is determined entirely by the 
        local neighborhood of the data point."
        """
        )
        st.subheader("The From-Scratch Python Code")
        st.info(
            "This simple implementation reveals the core 'brute-force' logic of k-NN: calculate all distances, find the closest, and take a vote."
        )

        code_string = inspect.getsource(SimpleKNNClassifier)
        st.code(code_string, language="python")

    st.markdown("---")

    # --- LEVEL 3: The Technical Deep Dive ---
    with st.expander("Technical Details & Trade-offs"):
        st.markdown(
            """
        #### **Our Implementation vs. Scikit-learn's**
        - **Our `SimpleKNNClassifier`:** Uses a **Brute-Force Search**. It calculates the 
        distance from the new point to *every single* training point. This is simple but 
        very slow on large datasets.
        - **Scikit-learn's `KNeighborsClassifier`:** Uses highly optimized data structures 
        like **Ball Trees** or **KD-Trees**. These structures partition the data, allowing 
        the algorithm to find the nearest neighbors extremely quickly without checking 
        every point.
            """
        )
