import streamlit as st
import pandas as pd
import joblib

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
    
    # This button must be indented to stay inside the Page 2 block
    if st.button("Calculate EMI Risk"):
        with st.spinner("Analyzing financial profile..."):
            try:
                # 1. Load Preprocessors and Models
                class_model = joblib.load('xgb_class_model.pkl')
                reg_model = joblib.load('xgb_reg_model.pkl')
                scaler = joblib.load('scaler.pkl')
                
                # 2. Build the Input DataFrame (All 32 Features)
                est_rent = salary * 0.20
                est_expenses = est_rent + 10000
                est_disposable = salary - est_expenses
                est_monthly_emi = loan_amount / 60 
                
                input_data = {
                    'age': age,
                    'gender': 0,               
                    'marital_status': 0,       
                    'education': 0,            
                    'monthly_salary': salary,
                    'employment_type': 0,      
                    'years_of_employment': 3.0,
                    'company_type': 0,         
                    'house_type': 0,           
                    'monthly_rent': est_rent,
                    'family_size': 3,
                    'dependents': 2,
                    'school_fees': 0,
                    'college_fees': 0,
                    'travel_expenses': 2000,
                    'groceries_utilities': 8000,
                    'other_monthly_expenses': 0,
                    'existing_loans': 0,
                    'current_emi_amount': 0,
                    'credit_score': credit_score,
                    'bank_balance': 25000,
                    'emergency_fund': 10000,
                    'emi_scenario': 0,         
                    'requested_amount': loan_amount,
                    'requested_tenure': 60,
                    'total_expenses': est_expenses,
                    'dti_ratio': est_expenses / (salary + 1),
                    'eti_ratio': est_monthly_emi / (salary + 1),
                    'disposable_income': est_disposable,
                    'affordability_ratio': est_disposable / (est_monthly_emi + 1),
                    'employment_stability_bonus': 1.0,
                    'custom_risk_score': credit_score * 0.8
                }
                
                input_df = pd.DataFrame([input_data])
                
               # 3. Transform and Predict
                
                # Ask the scaler exactly which columns it was trained on
                scaler_cols = scaler.feature_names_in_
                
                # Scale ONLY those specific columns in our dataframe
                input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
                
                # Ensure the columns are in the exact order the model expects
                model_cols = class_model.feature_names_in_
                input_df = input_df[model_cols]
                
                # Make the predictions using the properly formatted dataframe
                is_eligible = class_model.predict(input_df)[0]
                max_emi = reg_model.predict(input_df)[0]
                
                # 4. Display Live Results!
                st.write("---")
                st.subheader("📊 Assessment Results")
                
                if is_eligible == 1:
                    st.success("✅ **EMI Eligibility:** Eligible")
                else:
                    st.error("❌ **EMI Eligibility:** High Risk - Not Eligible")
                    
                st.info(f"💡 **Maximum Recommended EMI:** {max_emi:,.2f} INR / month")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ==========================================
# PAGE 3: Admin Dashboard
# ==========================================
elif page == "Admin Dashboard":
    st.title("⚙️ Admin & Model Monitoring")
    st.warning("This section is for administrative operations and MLflow tracking.")
