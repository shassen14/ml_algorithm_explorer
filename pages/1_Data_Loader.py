# pages/1_Data_Loader.py
import streamlit as st
import pandas as pd

from src.processing.data_manager import split_data

st.title("🖥️ Data Loader")

# Check if a problem type has been selected
if "problem_type" not in st.session_state or st.session_state.problem_type is None:
    st.warning("Please select a problem type on the Welcome page first.")
else:
    st.info(f"Current Problem Type: **{st.session_state.problem_type}**")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the DataFrame once, right here in the UI
            df = pd.read_csv(uploaded_file)

            st.subheader("Automated Data Cleaning")

            # 1. Handle customerID (or any unique identifier)
            # Drop unique identifier columns as they cause data leakage.
            # We identify them as object columns where every value is unique.
            id_cols = [
                col
                for col in df.select_dtypes(include="object").columns
                if df[col].nunique() == len(df)
            ]
            if id_cols:
                df = df.drop(columns=id_cols)
                st.success(
                    f"Removed identifier column(s) to prevent data leakage: `{'`, `'.join(id_cols)}`"
                )

            # 2. Handle TotalCharges (or other numeric-as-object columns)
            # Convert columns that look numeric but are stored as objects.
            for col in df.select_dtypes(include="object").columns:
                if col != st.session_state.get(
                    "target_column"
                ):  # Don't convert the target
                    try:
                        # Attempt to convert to numeric, coercing errors to NaN
                        converted_col = pd.to_numeric(df[col], errors="coerce")
                        # If conversion is successful for a good portion of the data, apply it
                        if (
                            converted_col.notna().sum() / len(df) > 0.8
                        ):  # Heuristic: if >80% are numeric
                            df[col] = converted_col
                            st.success(
                                f"Successfully converted column `{col}` to a numeric type."
                            )
                    except Exception:
                        continue  # Ignore columns that can't be converted

            st.write("Data Preview (after cleaning):")
            st.dataframe(df.head())

            # Store the full dataframe in session state for EDA page
            st.session_state["full_df"] = df

            target_column = st.selectbox("Select the target column", df.columns)
            test_size = st.slider("Select the test set size", 0.1, 0.5, 0.2)

            if st.button("Process and Split Data"):
                # Store the selected target column in the session state
                st.session_state["target_column"] = target_column

                with st.spinner("Processing..."):
                    # Backend function with the DataFrame
                    X_train, X_test, y_train, y_test = split_data(
                        df, target_column, test_size
                    )

                    if X_train is not None:
                        st.success("Data successfully processed and split!")

                        st.session_state["processed_data"] = {
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                        }

                        st.subheader("Data Split Summary")
                        st.write(f"**Training Features Shape:** `{X_train.shape}`")
                        st.write(f"**Testing Features Shape:** `{X_test.shape}`")
                        st.write(f"**Training Target Shape:** `{y_train.shape}`")
                        st.write(f"**Testing Target Shape:** `{y_test.shape}`")
                    else:
                        st.error(
                            "Failed to process the data. Check the terminal for logs."
                        )

        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
