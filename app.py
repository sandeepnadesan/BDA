import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('loan_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict(data):
    # Make sure the features passed to the model match the trained feature set (11 features)
    features = np.array([[
        data['Gender'],
        data['Married'],
        data['Dependents'],
        data['Education'],
        data['Self_Employed'],
        data['Property_Area'],
        data['LoanAmount'],
        data['Loan_Amount_Term'],
        data['Credit_History'],
        data['ApplicantIncome'],
        data['CoapplicantIncome']
    ]])
    prediction = model.predict(features)
    return "Loan Approved" if prediction == 1 else "Loan Denied"

# Streamlit UI Setup
def main():
    st.title("Loan Prediction")

    # Input form for user data
    gender = st.selectbox("Gender", ("Male", "Female"))
    married = st.selectbox("Married", ("No", "Yes"))
    dependents = st.number_input("Dependents", min_value=0, max_value=5, step=1)
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.selectbox("Self Employed", ("No", "Yes"))
    property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    # Additional fields (missing in the original form)
    loan_amount = st.number_input("Loan Amount", min_value=0, max_value=1000000, step=1000)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, max_value=480, step=1)
    credit_history = st.selectbox("Credit History", (0, 1))  # Assuming 0 or 1 for credit history
    applicant_income = st.number_input("Applicant Income", min_value=0, max_value=10000000, step=1000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, max_value=10000000, step=1000)

    # Map categorical inputs to numerical values as the model expects
    gender = 1 if gender == "Female" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Not Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 0, "Semiurban": 1, "Rural": 2}[property_area]
    
    # Display input data summary
    st.write("### Input Data:")
    st.write(f"Gender: {gender}, Married: {married}, Dependents: {dependents}, Education: {education}, Self Employed: {self_employed}, Property Area: {property_area}")
    st.write(f"Loan Amount: {loan_amount}, Loan Amount Term: {loan_amount_term}, Credit History: {credit_history}, Applicant Income: {applicant_income}, Coapplicant Income: {coapplicant_income}")

    # Button to trigger the prediction
    if st.button("Predict"):
        # Make prediction
        data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income
        }
        prediction = predict(data)

        # Display the result
        st.write(f"### Prediction: {prediction}")

if __name__ == "__main__":
    main()
