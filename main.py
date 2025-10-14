# main.py
import streamlit as st

# Set the page config once, here
st.set_page_config(page_title="ML Algorithm Explorer", page_icon="🤖", layout="wide")

# Define the pages
pg = st.navigation(
    {
        "Welcome": [st.Page("pages/0_Welcome.py", title="Introduction", icon="🏠")],
        "Analysis Tools": [
            st.Page("pages/1_Data_Loader.py", title="Data Loader", icon="🖥️"),
            st.Page("pages/2_Data_Preparation.py", title="Data Preparation", icon="🛠️"),
            st.Page("pages/2_EDA.py", title="EDA", icon="📊"),
            st.Page("pages/3_Model_Explorer.py", title="Model Explorer", icon="🔬"),
        ],
    }
)

# Run the navigation
pg.run()
