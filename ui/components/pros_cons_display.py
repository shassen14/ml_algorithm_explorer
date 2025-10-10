# ui/components/pros_cons_display.py
import streamlit as st
from src.content.algorithm_descriptions import ALGORITHM_INFO


def render(model_name):
    """
    Renders a structured Pros & Cons section for a given model.
    """
    st.markdown("---")  # Add a separator
    with st.expander(f"View General Trade-offs of {model_name}"):

        # Look up the information for the selected model
        info = ALGORITHM_INFO.get(model_name)

        if not info:
            st.warning("No description available for this model yet.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Pros")
            # Use markdown to render list items
            for pro in info.get("pros", []):
                st.markdown(f"- {pro}")

        with col2:
            st.markdown("#### ❌ Cons")
            for con in info.get("cons", []):
                st.markdown(f"- {con}")
