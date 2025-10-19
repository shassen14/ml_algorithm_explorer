# ui/components/transformer_ui.py
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import json
from src.processing.preparation_manager import DataPreparationManager
from src.schemas import (
    MapKeywordsParams,
    TransformationStep,
    DropColumnsParams,
    RenameColumnParams,
    ConvertTypeParams,
    MapValuesParams,
    MathOperationParams,
    ExtractTextParams,
    BooleanFlagParams,
    FillNaParams,
)
from typing import List


# ==============================================================================
# 1. SIDEBAR COMPONENT
# ==============================================================================
def render_sidebar(manager: DataPreparationManager, df: pd.DataFrame) -> str:
    """
    Renders the sidebar for column selection and recipe management.

    Args:
        df (pd.DataFrame): The current working dataframe.
        recipe (List[TransformationStep]): The list of current transformation steps.

    Returns:
        str: The name of the column selected by the user.
    """
    st.sidebar.header("Data Transformer Workbench")

    st.sidebar.markdown("##### 1. Select a Column to Inspect")
    columns = df.columns.tolist()
    selected_column = st.sidebar.radio(
        "Columns", options=columns, label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 2. Transformation Recipe")

    if not manager.recipe.steps:
        st.sidebar.info("Your recipe is empty.")
    else:
        # Loop in reverse for stable delete/move operations
        for i, step in reversed(list(enumerate(manager.recipe.steps))):
            col1, col2, col3, col4 = st.sidebar.columns([4, 1, 1, 1])
            col1.text(f"{i+1}. {step.step_type.replace('_', ' ').title()}")
            col2.button(
                "🔼",
                key=f"up_{i}",
                help="Move step up",
                on_click=manager.move_step,
                args=(i, "up"),
            )
            col3.button(
                "🔽",
                key=f"down_{i}",
                help="Move step down",
                on_click=manager.move_step,
                args=(i, "down"),
            )
            col4.button(
                "🗑️",
                key=f"del_{i}",
                help="Delete step",
                on_click=manager.remove_step,
                args=(i,),
            )

    if st.sidebar.button("Reset Recipe"):
        manager.reset_recipe()
        st.rerun()

    return selected_column


# ==============================================================================
# 2. MAIN PANEL COMPONENTS
# ==============================================================================
def render_column_inspector(
    manager: DataPreparationManager, df: pd.DataFrame, selected_column: str
):
    """
    Renders the main inspection panel for a single selected column,
    including univariate analysis and available in-place transformations.
    """
    st.header(f"Inspecting Column: `{selected_column}`")

    col_data = df[selected_column]

    # --- Display Missing Value Info ---
    missing_count = col_data.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    st.metric("Missing Values", f"{missing_count} ({missing_percent:.2f}%)")

    # --- Univariate Analysis ---
    st.subheader("Analysis & Statistics")
    if pd.api.types.is_numeric_dtype(col_data):
        st.metric("Mean", f"{col_data.mean():.2f}")
        fig, ax = plt.subplots()
        sns.histplot(col_data, kde=True, ax=ax)
        st.pyplot(fig)
    else:  # Categorical
        st.metric("Unique Values", col_data.nunique())
        st.write("Value Counts (Top 20):")
        st.dataframe(col_data.value_counts().head(20))

    st.markdown("---")
    st.subheader("Univariate Transformations (Acts on this column)")

    # --- In-place Transformations ---
    with st.expander("Rename Column"):
        with st.form(f"rename_{selected_column}"):
            new_name = st.text_input("New column name", value=selected_column)
            if st.form_submit_button("Add to Recipe"):
                params = RenameColumnParams(
                    source_column=selected_column, new_name=new_name
                )
                step = TransformationStep(step_type="rename_column", params=params)
                manager.add_step(step)
                st.rerun()

    if pd.api.types.is_object_dtype(col_data):
        with st.expander("Convert to Numeric"):
            st.info(
                "This will attempt to extract numbers from the text (e.g., '8GB' -> 8). Non-numeric parts will be ignored."
            )
            if st.button(
                "Add 'Convert to Numeric' to Recipe", key=f"convert_{selected_column}"
            ):
                params = ConvertTypeParams(
                    column=selected_column, target_type="numeric"
                )
                st.session_state.recipe.steps.append(
                    TransformationStep(step_type="convert_type_numeric", params=params)
                )
                st.rerun()

    with st.expander("Map & Group Values"):
        with st.form(f"map_{selected_column}"):
            st.info(
                "Create a new column by mapping values from the selected column. Unmapped values will be kept as-is unless a default is provided."
            )
            new_col_name = st.text_input(
                "Name for new mapped column", f"{selected_column}_mapped"
            )
            mapping_json = st.text_area(
                "Mapping dictionary (JSON format)",
                height=150,
                placeholder='{"Old Value 1": "New Value", ...}',
            )
            default_val = st.text_input(
                "Default for unmapped values (optional)", placeholder="e.g., Other"
            )
            if st.form_submit_button("Add to Recipe"):
                try:
                    mapping_dict = json.loads(mapping_json)
                    params = MapValuesParams(
                        column=selected_column,
                        new_column_name=new_col_name,
                        mapping_dict=mapping_dict,
                        default_value=default_val or None,
                    )
                    step = TransformationStep(step_type="map_values", params=params)
                    manager.add_step(step)
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in mapping dictionary.")
    with st.expander("Fill Missing Values (Impute)"):
        with st.form(f"fillna_{selected_column}"):
            st.info("Choose a strategy to fill the missing values in this column.")

            # Allow different strategies based on column type
            if pd.api.types.is_numeric_dtype(col_data):
                strategy = st.selectbox(
                    "Imputation Strategy:", ["Specific Value", "Mean", "Median", "Mode"]
                )
            else:
                strategy = st.selectbox(
                    "Imputation Strategy:", ["Specific Value", "Mode"]
                )

            fill_val = None
            if strategy == "Specific Value":
                fill_val = st.text_input("Value to use for filling", value="0")

            if st.form_submit_button("Add to Recipe"):
                params = FillNaParams(
                    column=selected_column,
                    strategy=strategy.lower().replace(
                        " ", "_"
                    ),  # e.g., "Specific Value" -> "specific_value"
                    fill_value=fill_val,
                )
                step = TransformationStep(step_type="fill_na", params=params)
                manager.add_step(step)
                st.rerun()


def render_feature_workbench(manager: DataPreparationManager, df: pd.DataFrame):
    """
    Renders the feature engineering workbench for creating new columns
    from one or more existing columns (multivariate transformations).
    """
    st.markdown("---")
    st.header("🛠️ Feature Engineering Workbench")

    with st.expander("Create New Features"):
        with st.form("boolean_flag_form"):
            st.markdown("###### Create Boolean Flag from Keyword")
            st.info(
                "Creates a new column with 1 if a keyword is found in a text column, and 0 otherwise."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                source_col = st.selectbox(
                    "Source column",
                    options=df.select_dtypes(include="object").columns,
                    key="bool_source",
                )
            with col2:
                keyword = st.text_input(
                    "Keyword to find", placeholder="e.g., Touchscreen"
                )
            with col3:
                new_col_name = st.text_input(
                    "New column name", placeholder="e.g., Is_Touchscreen"
                )

            case_sensitive = st.checkbox("Case sensitive search")

            if st.form_submit_button("Add to Recipe"):
                if source_col and keyword and new_col_name:
                    params = BooleanFlagParams(
                        source_column=source_col,
                        keyword=keyword,
                        new_column_name=new_col_name,
                        case_sensitive=case_sensitive,
                    )
                    step = TransformationStep(
                        step_type="create_boolean_flag", params=params
                    )
                    manager.add_step(step)
                    st.rerun()

        with st.form("math_op_form"):
            st.markdown("###### Create Column with a Math Formula")
            new_col_name = st.text_input("New column name")

            formula = st.text_input(
                "Formula",
                help="Enter a Python/Pandas expression. Use column names as variables. `np` is available for numpy functions.",
            )
            st.info(
                """
            **Valid Formula Examples:**
            - `2024 - year`
            - `revenue - cost`
            - `np.log1p(price)`
            - `np.where(temp_hdd_unit == 'TB', temp_hdd_num * 1024, temp_hdd_num)`
            - `(col_A + col_B) / 2`
            """
            )

            if st.form_submit_button("Add to Recipe"):
                if new_col_name and formula:
                    params = MathOperationParams(
                        new_column_name=new_col_name, formula=formula
                    )
                    step = TransformationStep(step_type="math_operation", params=params)
                    manager.add_step(step)
                    st.rerun()

        with st.form("regex_extract_form"):
            st.markdown("###### Create Column by Extracting Text (Regex)")
            source_col = st.selectbox("Source column", options=df.columns)
            new_col_name_re = st.text_input("New column name", key="re_new_name")
            regex_pattern = st.text_input(
                "Regex pattern",
                help="Use parentheses `()` to create a capture group. e.g., `(Intel|AMD)`",
            )
            if st.form_submit_button("Add to Recipe"):
                if source_col and new_col_name_re and regex_pattern:
                    params = ExtractTextParams(
                        source_column=source_col,
                        new_column_name=new_col_name_re,
                        regex_pattern=regex_pattern,
                    )
                    step = TransformationStep(
                        step_type="extract_text_regex", params=params
                    )
                    manager.add_step(step)
                    st.rerun()
        with st.form("map_keywords_form"):
            st.markdown("###### Create Column by Mapping Keywords")
            st.info(
                "Searches for keywords in a source column and assigns a new value. Perfect for grouping categories."
            )

            source_col = st.selectbox(
                "Source Column", options=df.columns, key="map_kw_source"
            )
            new_col_name = st.text_input("New Column Name", key="map_kw_new_name")

            mapping_text = st.text_area(
                "Keyword Mapping (one per line: `keyword -> new_value`)",
                height=200,
                placeholder="Core i7 -> Intel High-End\nCore i5 -> Intel Mid-Range\nRyzen 7 -> AMD High-End",
            )

            default_val = st.text_input("Default for unmatched values", value="Other")

            if st.form_submit_button("Add to Recipe"):
                try:
                    # Parse the text area input into a dictionary
                    mapping_dict = {}
                    for line in mapping_text.strip().split("\n"):
                        if "->" in line:
                            keyword, new_value = line.split("->", 1)
                            mapping_dict[keyword.strip()] = new_value.strip()

                    if source_col and new_col_name and mapping_dict:
                        params = MapKeywordsParams(
                            source_column=source_col,
                            new_column_name=new_col_name,
                            keyword_mapping=mapping_dict,
                            default_value=default_val,
                        )
                        step = TransformationStep(
                            step_type="map_by_keywords", params=params
                        )
                        manager.add_step(step)
                        st.rerun()
                except Exception as e:
                    st.error(
                        f"Could not parse mapping text. Please check the format. Error: {e}"
                    )

        with st.form("drop_cols_form"):
            st.markdown("###### Drop Multiple Columns")
            cols_to_drop = st.multiselect("Select columns to drop", options=df.columns)
            if st.form_submit_button("Add to Recipe", type="secondary"):
                if cols_to_drop:
                    params = DropColumnsParams(columns=cols_to_drop)
                    step = TransformationStep(step_type="drop_columns", params=params)
                    manager.add_step(step)
                    st.rerun()
