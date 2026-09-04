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
                import joblib
                import pandas as pd
                
                # 1. Load Preprocessors and Models
                class_model = joblib.load('xgb_class_model.pkl')
                reg_model = joblib.load('xgb_reg_model.pkl')
                scaler = joblib.load('scaler.pkl')
                
                # 2. Build the Input DataFrame 
                # IMPORTANT: You must update this dictionary to match the EXACT column names 
                # and exact number of columns (e.g., 22) that your X_train dataset had!
                input_data = {
                    'age': age,
                    'monthly_salary': salary,
                    'credit_score': credit_score,
                    'loan_amount': loan_amount,
                    # --- ADD YOUR MISSING COLUMNS BELOW WITH DEFAULT VALUES ---
                    # 'dependents': 2,
                    # 'existing_loans': 0,
                    # 'dti_ratio': loan_amount / (salary + 1), # Engineered feature!
                    # ... add the rest of your X_train columns here ...
                }
                
                input_df = pd.DataFrame([input_data])
                
                # 3. Transform and Predict
                input_scaled = scaler.transform(input_df)
                
                is_eligible = class_model.predict(input_scaled)[0]
                max_emi = reg_model.predict(input_scaled)[0]
                
                # 4. Display Live Results!
                st.write("---")
                st.subheader("📊 Assessment Results")
                
                if is_eligible == 1:
                    st.success("✅ **EMI Eligibility:** Eligible")
                else:
                    st.error("❌ **EMI Eligibility:** High Risk - Not Eligible")
                    
                st.info(f"💡 **Maximum Recommended EMI:** {max_emi:,.2f} INR / month")
                
            except ValueError as ve:
                st.error(f"Data Mismatch Error: {ve}")
                st.warning("Hint: Make sure the 'input_data' dictionary in your code has the exact same columns as your training data!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
# ==========================================
# PAGE 3: Admin Dashboard
# ==========================================
elif page == "Admin Dashboard":
    st.title("⚙️ Admin & Model Monitoring")
    st.warning("This section is for administrative operations and MLflow tracking.")
