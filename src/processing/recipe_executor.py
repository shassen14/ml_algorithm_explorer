# src/processing/recipe_executor.py
import pandas as pd
from src.schemas import TransformationRecipe
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_recipe(raw_df: pd.DataFrame, recipe: TransformationRecipe) -> pd.DataFrame:
    """Applies a series of transformation steps to a raw dataframe."""
    df = raw_df.copy()

    for i, step in enumerate(recipe.steps):
        if not step.is_active:
            continue
        try:
            step_type = step.step_type
            params = step.params

            if step_type == "drop_columns":
                df = df.drop(columns=params.columns, errors="ignore")
            elif step_type == "rename_column":
                df = df.rename(columns={params.source_column: params.new_name})
            elif step_type == "convert_type_numeric":
                df[params.column] = pd.to_numeric(
                    df[params.column]
                    .astype(str)
                    .str.extract(r"(\d+\.?\d*)", expand=False),
                    errors="coerce",
                )
            elif step_type == "map_values":
                if params.default_value is not None:
                    df[params.new_column_name] = (
                        df[params.column]
                        .map(params.mapping_dict)
                        .fillna(params.default_value)
                    )
                else:
                    df[params.new_column_name] = df[params.column].replace(
                        params.mapping_dict
                    )
            elif step_type == "extract_text_regex":
                df[params.new_column_name] = (
                    df[params.source_column]
                    .astype(str)
                    .str.extract(params.regex_pattern)
                )
            elif step_type == "math_operation":
                params = step.params
                formula = params.formula
                new_col_name = params.new_column_name

                logger.info(
                    f"Applying math operation to create column '{new_col_name}' with formula: '{formula}'"
                )

                # 1. Create a "safe" local namespace for the evaluation.
                # This dictionary will contain all the variables the formula is allowed to see.
                eval_namespace = {
                    "np": np,
                    "pd": pd,
                    # Add other safe libraries here if needed
                }
                # 2. Add all existing dataframe columns to the namespace as Pandas Series.
                # This allows the formula to reference columns directly by name (e.g., `year` or `temp_hdd_num`).
                for col in df.columns:
                    eval_namespace[col] = df[col]

                # 3. Use Python's built-in eval() with the carefully constructed namespace.
                # The 'globals' is empty to prevent access to anything outside our safe namespace.
                result_series = eval(formula, {"__builtins__": {}}, eval_namespace)

                # 4. Assign the resulting series to the new column.
                df[new_col_name] = result_series

                logger.info(f"Successfully created column '{new_col_name}'.")
            elif step.step_type == "create_boolean_flag":
                params = step.params
                # Use .str.contains() which is perfect for this.
                # `case=False` makes it case-insensitive by default.
                # `na=False` ensures that missing values in the source become 0 (False), not an error.
                df[params.new_column_name] = (
                    df[params.source_column]
                    .astype(str)
                    .str.contains(
                        params.keyword, case=not params.case_sensitive, na=False
                    )
                    .astype(int)
                )  # Convert the True/False result to 1/0
            elif step.step_type == "map_by_keywords":
                params = step.params
                source_col = df[params.source_column].astype(str)

                # Start with the default value for the new column
                new_col = pd.Series([params.default_value] * len(df), index=df.index)

                # Iteratively apply the mapping. The order matters if keywords overlap.
                # A more robust implementation might use np.select.
                for keyword, new_value in params.keyword_mapping.items():
                    # case=False makes the search case-insensitive
                    new_col[source_col.str.contains(keyword, case=False, na=False)] = (
                        new_value
                    )

                df[params.new_column_name] = new_col

        except Exception as e:
            raise ValueError(
                f"Recipe failed at Step {i+1} ('{step_type}'). Details: {e}"
            )

    return df


def generate_python_script(recipe: TransformationRecipe) -> str:
    """Generates a Python script string from a transformation recipe."""
    script_lines = [
        "import pandas as pd",
        "import numpy as np",
        "\ndef transform_data(df: pd.DataFrame) -> pd.DataFrame:",
        "    # This script was auto-generated by the ML Explorer App",
    ]

    for step in recipe.steps:
        if not step.is_active:
            script_lines.append(f"\n    # Step: {step.step_type} (DISABLED)")
            continue

        step_type = step.step_type
        params = step.params
        script_lines.append(f"\n    # Step: {step_type}")

        if step_type == "drop_columns":
            script_lines.append(
                f"    df = df.drop(columns={params.columns}, errors='ignore')"
            )
        elif step_type == "rename_column":
            script_lines.append(
                f"    df = df.rename(columns={{'{params.source_column}': '{params.new_name}'}})"
            )
        # ... Add similar logic for other step types ...
        elif step_type == "math_operation":
            formula_str = params.formula.replace("{", "df['").replace("}", "']")
            script_lines.append(f"    df['{params.new_column_name}'] = {formula_str}")

    script_lines.append("\n    return df")
    return "\n".join(script_lines)
