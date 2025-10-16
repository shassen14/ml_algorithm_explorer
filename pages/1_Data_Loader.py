# pages/1_Data_Loader.py
import streamlit as st
import pandas as pd
import logging

from src.analysis.profiler import profile_dataframe

logger = logging.getLogger(__name__)

st.title("💾 Data Loader")
st.markdown(
    "Upload a CSV file to begin your analysis. This is the first step in the workflow."
)

# --- 1. Check if a problem type has been selected from the Welcome page ---
if "problem_type" not in st.session_state or st.session_state.problem_type is None:
    st.warning("Please select a problem type on the 🏠 Welcome page first.")
    st.stop()  # Halt execution until a problem type is chosen

st.info(f"Current Problem Type: **{st.session_state.problem_type}**")

# --- 2. File Uploader and Advanced Encoding Options ---
# List of common encodings to try
COMMON_ENCODINGS = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="If you see an error after uploading, your file might have a non-standard encoding.",
)

with st.expander("Advanced Options"):
    selected_encoding = st.selectbox("File Encoding:", COMMON_ENCODINGS)

# --- 3. Intelligent Loading Logic ---
if uploaded_file is not None:

    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        if "last_run_result" in st.session_state:
            del st.session_state["last_run_result"]
        st.session_state["uploaded_file_name"] = uploaded_file.name

    # Store the file in session state to avoid reloading on every rerun
    st.session_state["uploaded_file"] = uploaded_file

    df = None
    try:
        # We use a key for the file uploader to help Streamlit manage state
        uploaded_file.seek(0)  # Reset file buffer to the beginning
        logger.info(f"Attempting to read CSV with encoding: {selected_encoding}")
        df = pd.read_csv(uploaded_file, encoding=selected_encoding)
        st.success(f"Successfully loaded the file with '{selected_encoding}' encoding.")

    except UnicodeDecodeError as e:
        st.error(
            f"Error reading file with '{selected_encoding}' encoding. This is common for non-standard files."
        )
        st.info(
            "💡 **Tip:** Try a different encoding from the 'Advanced Options' dropdown, such as 'latin1'."
        )
        logger.error(f"UnicodeDecodeError with {selected_encoding}: {e}")
        st.stop()

    except Exception as e:
        st.error(f"An unexpected error occurred while reading the file: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        st.stop()

    # --- 4. Store Raw Data and Direct the User ---
    if df is not None:
        # Store the raw, unmodified dataframe for the next step
        st.session_state["raw_df"] = df

        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df.head())

        st.subheader("Automated Data Profile")
        with st.spinner("Profiling data..."):
            insights = profile_dataframe(df)
            for insight in insights:
                if insight["type"] == "info":
                    st.info(insight["text"], icon="ℹ️")
                elif insight["type"] == "warning":
                    st.warning(insight["text"], icon="⚠️")
                elif insight["type"] == "error":
                    st.error(insight["text"], icon="❗️")

        st.markdown("---")
        st.success(
            "✅ Data loaded! Please proceed to the **🛠️ Data Transformer** page to clean, prepare, and split your data for modeling."
        )

        # Clear out any old processed data to prevent state conflicts
        if "processed_data" in st.session_state:
            del st.session_state["processed_data"]
        if "full_df" in st.session_state:
            del st.session_state["full_df"]
