import streamlit as st
import pandas as pd
import pickle as pk

feature_columns = [
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'net_worth'
]

result_placeholder = st.empty()

from sklearn.preprocessing import MinMaxScaler

model =  pk.load(open(r'C:\Users\Naresh\PycharmProjects\Loan-approval-ML\models\model.pkl', 'rb'))
scaler = pk.load(open(r'C:\Users\Naresh\PycharmProjects\Loan-approval-ML\models\scaler.pkl', 'rb'))


st.header('Loan Prediction Web App')

with st.form("loan_form"):
    no_of_dep = st.slider("Choose No of dependents", 0, 5)
    grad = st.selectbox("Choose Education", ["Graduated", "Not Graduated"])
    self_emp = st.selectbox("Self employed", ["Yes", "No"])
    annual_income = st.slider("Annual Income", 0, 10000000)
    loan_amt = st.slider("Loan Amount", 0, 10000000)
    loan_duration = st.slider("Loan Duration", 1, 60)
    cibil = st.slider("Enter your CIBIL Score", 0, 900)
    assets = st.slider("Choose your net worth", 0, 1000000000)

    submitted = st.form_submit_button("Predict")





if submitted:
    grad_s = 1 if grad == "Graduated" else 0
    emp_s = 1 if self_emp == "Yes" else 0

    pred_data = pd.DataFrame(
        [[
            no_of_dep,
            grad_s,
            emp_s,
            annual_income,
            loan_amt,
            loan_duration,
            cibil,
            assets
        ]],
        columns=feature_columns
    )


    pred_data = pd.DataFrame(
        [[
            no_of_dep,
            grad_s,
            emp_s,
            annual_income,
            loan_amt,
            loan_duration,
            cibil,
            assets
        ]],
        columns=feature_columns
    )

    pred_data_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_data_scaled)

    result_placeholder.empty()

    pred_data_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_data_scaled)

    if prediction[0] == 1:
        st.success("Loan is Approved")
    else:
        st.error("Loan is Not Approved")


