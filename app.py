import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Smart Retail Demand Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("Navigation")
    st.info("Project setup phase initialized.")
    st.markdown("---")
    st.write("### About")
    st.write(
        "AI-powered retail demand forecasting system designed to optimize inventory, "
        "predict consumer demand, and minimize stockouts and overstocking."
    )

# Main Page Header
st.title("Smart Retail Demand Forecasting System")
st.markdown("Welcome to the **Smart Retail Demand Forecasting Platform**.")

st.markdown("---")

# Welcome & Overview Section
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Status", value="Initialized", delta="Ready")
    st.markdown("##### 📊 Data Analysis")
    st.caption("Inspect historical sales, seasonal trends, and promotional patterns.")

with col2:
    st.metric(label="Forecasting Engine", value="Pending Setup")
    st.markdown("##### 🤖 ML Forecasting Models")
    st.caption("Leverage XGBoost, LightGBM, and time-series algorithms for accurate demand forecasts.")

with col3:
    st.metric(label="Inventory Optimization", value="Planned")
    st.markdown("##### 📦 Stock & Reorder Insights")
    st.caption("Translate demand forecasts into actionable inventory replenishment decisions.")

st.markdown("---")

# Getting Started Guide
st.subheader("Getting Started")
st.markdown(
    """
    The initial project scaffolding has been created successfully.
    
    **Next Steps:**
    1. Place historical retail data in the `data/` directory.
    2. Explore and preprocess data using notebooks in `notebooks/`.
    3. Implement data loaders, preprocessing, and feature engineering inside `src/`.
    4. Train and save models in `models/`.
    """
)
