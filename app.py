import streamlit as st
import pandas as pd

# Set the configuration for the web page
st.set_page_config(page_title="EMIPredict AI", layout="wide")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home - Data Explorer", "EMI Prediction", "Admin Dashboard"])

# ==========================================
# PAGE 1: Home
# ==========================================
if page == "Home - Data Explorer":
    st.title("📊 EMIPredict AI - Financial Risk Platform")
    st.markdown("Welcome to the Intelligent Financial Risk Assessment Platform.")
    st.info("Navigate to the 'EMI Prediction' tab to assess your loan eligibility.")

# ==========================================
# PAGE 2: EMI Prediction (The Core Feature)
# ==========================================
elif page == "EMI Prediction":
    st.title("💰 Real-Time EMI Prediction")
    st.write("Enter the applicant's details below to assess EMI risk and maximum affordability.")
    
    # Create two columns for user input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        
    with col2:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Requested Loan Amount", min_value=0, value=500000)
        scenario = st.selectbox("EMI Scenario", ["Personal Loan", "Vehicle", "Home Appliances", "Education", "E-commerce"])
    
    if st.button("Calculate EMI Risk"):
        with st.spinner("Analyzing financial profile..."):
            try:
                # 1. Connect to your local MLflow server
                import mlflow
                mlflow.set_tracking_uri("http://localhost:5000")
                
                # 2. Load the best models from the registry
                # (Note: In a fully finished app, you would transform the user inputs here using your saved StandardScaler)
                class_model = mlflow.xgboost.load_model("models:/EMI_Classification_Model/Latest")
                reg_model = mlflow.xgboost.load_model("models:/EMI_Regression_Model/Latest")
                
                # 3. Display the results!
                st.write("---")
                st.subheader("📊 Assessment Results")
                
                # We will use mock results here temporarily until we add the data scaling logic
                st.success("✅ **EMI Eligibility:** Eligible")
                st.info(f"💡 **Maximum Recommended EMI:** 12,500 INR / month")
                
                st.write("**Note:** To connect the live model, we must add the StandardScaler and Feature Engineering logic to transform the UI inputs!")
                
            except Exception as e:
                st.error(f"Could not connect to MLflow models. Make sure the MLflow server is running! Error: {e}")

# ==========================================
# PAGE 3: Admin Dashboard
# ==========================================
elif page == "Admin Dashboard":
    st.title("⚙️ Admin & Model Monitoring")
    st.warning("This section is for administrative operations and MLflow tracking.")