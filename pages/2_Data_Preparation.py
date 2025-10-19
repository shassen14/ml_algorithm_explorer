# pages/2_Data_Preparation.py
import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np

# Import our custom schemas, the backend transformer engine, and the UI components
from src.processing.preparation_manager import DataPreparationManager
from src.schemas import TransformationRecipe
from src.processing.recipe_executor import generate_python_script
from ui.components import preparation_ui

# --- 1. Page Configuration and Initial Setup ---
st.set_page_config(layout="wide")
st.title("🛠️ Data Transformer")
st.info(
    """
This is your interactive workbench for data preparation. Select a column from the sidebar to inspect and apply **Univariate Transformations**. 
Use the **Feature Engineering Workbench** below the data preview to create new columns from one or more existing columns. 
Your steps are recorded in the **Transformation Recipe** in the sidebar.
"""
)

# --- 2. Initialization and State Management ---
# Prerequisite check: ensure raw data from the loader exists.
if "raw_df" not in st.session_state:
    st.warning("Please load data on the 💾 Data Loader page first.")
    st.stop()

# Initialize the recipe object in the session state if it doesn't exist.
if "recipe" not in st.session_state:
    st.session_state.recipe = TransformationRecipe()


# --- 3. The Core Non-Destructive Transformation Engine ---
# Create the manager instance on every rerun, injecting the persistent recipe
# object from st.session_state.
manager = DataPreparationManager(
    raw_df=st.session_state.raw_df,
    recipe=st.session_state.recipe,
)
working_df = manager.get_working_df()


# --- 4. Render Layout and UI Components ---

# The sidebar component handles column selection and recipe management.
# It returns the name of the column currently selected by the user.
selected_column = preparation_ui.render_sidebar(manager, working_df)


# The main area is split into the data preview and the transformation workbenches.
st.subheader("Current Data Preview")
st.dataframe(working_df.head(100))
st.write(f"**Current Shape:** {working_df.shape}")

# The column inspector component renders analysis and actions for the selected column.
preparation_ui.render_column_inspector(manager, working_df, selected_column)

# The feature workbench component renders tools for creating new columns.
preparation_ui.render_feature_workbench(manager, working_df)


# --- 5. Finalization, Splitting, and Exporting ---
st.markdown("---")
with st.expander("✅ Export & Finalize Data for Modeling", expanded=True):

    col1, col2 = st.columns([1, 2])  # Give more space to the finalization logic

    # --- EXPORTING CONTROLS ---
    with col1:
        st.markdown("##### Download Artifacts")
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=working_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Recipe (JSON)",
            data=st.session_state.recipe.model_dump_json(indent=2),
            file_name="recipe.json",
            mime="application/json",
        )
        st.download_button(
            label="Download Python Script",
            data=generate_python_script(st.session_state.recipe),
            file_name="transform_script.py",
            mime="text/x-python",
        )

    # --- FINALIZATION AND SPLITTING CONTROLS ---
    with col2:
        st.markdown("##### Finalize for Modeling")
        if working_df.empty:
            st.error("Cannot finalize an empty dataframe. Please adjust your recipe.")
        else:
            target_column = st.selectbox(
                "Select the final target column:",
                working_df.columns,
                index=max(0, len(working_df.columns) - 1),  # Default to the last column
            )

            if st.session_state.get("problem_type") == "Regression":
                transform_method = st.selectbox(
                    "Select a Target Variable Transformation:",
                    options=[
                        "None",
                        "Log Transform (log1p)",
                        "Square Root Transform",
                        "Box-Cox Transform",
                    ],
                    help="Transforms the target variable to stabilize variance and handle "
                    "skewed data. The model's results will be automatically inverse-transformed "
                    "for interpretation.",
                )
            else:
                transform_method = "None"

            test_size = st.slider("Select the test set size (%):", 10, 50, 20) / 100.0

            if st.button("Finalize and Split Data", type="primary"):
                with st.spinner("Processing..."):
                    final_df = working_df.copy()

                    # Store the chosen method in session state for the pipeline
                    st.session_state["target_transform_method"] = transform_method

                    final_df.dropna(subset=[target_column], inplace=True)

                    X = final_df.drop(columns=[target_column])
                    y = final_df[target_column]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    st.session_state["processed_data"] = {
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                    }
                    st.session_state["full_df"] = final_df  # For EDA page
                    st.session_state["target_column"] = target_column

                    st.success("Data has been successfully transformed and split!")
                    st.info("Please proceed to the **🔬 Model Explorer** page.")
