"""Loan Approval Prediction - Streamlit UI.   Run:  streamlit run app.py"""
import os

import joblib
import pandas as pd
import streamlit as st

import loan_features  # noqa: F401  (needed to unpickle the pipeline)

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "loan_model.joblib"))


model = load_model()
st.title("🏦 Loan Approval Prediction System")
st.caption("Enter applicant details. The model estimates the probability that the loan is approved.")

with st.form("loan"):
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    married = c2.selectbox("Marital status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Single")
    dependents = c1.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = c2.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = c1.selectbox("Self employed", ["No", "Yes"])
    area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    income = c1.number_input("Applicant income (Rs / month)", 1000, 200000, 6000, 500)
    co_income = c2.number_input("Co-applicant income (Rs / month)", 0, 100000, 0, 500)
    loan = c1.number_input("Loan amount (Rs thousand)", 10, 1000, 150, 10)
    term = c2.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 84, 480], index=0)
    credit = st.radio("Credit history meets guidelines?", [1.0, 0.0], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    go = st.form_submit_button("Predict", type="primary")

if go:
    X = pd.DataFrame([dict(Gender=gender, Married=married, Dependents=dependents, Education=education,
                           Self_Employed=self_emp, ApplicantIncome=income, CoapplicantIncome=co_income,
                           LoanAmount=loan, Loan_Amount_Term=term, Credit_History=credit, Property_Area=area)])
    p = float(model.predict_proba(X)[0, 1])
    emi = loan * 1000 / term
    ratio = emi / (income + co_income)
    st.divider()
    if p >= 0.5:
        st.success(f"✅ Likely APPROVED  (probability {p:.0%})")
    else:
        st.error(f"❌ Likely REJECTED  (approval probability {p:.0%})")
    st.progress(p)
    m1, m2 = st.columns(2)
    m1.metric("Estimated monthly instalment", f"Rs {emi:,.0f}")
    m2.metric("Instalment / income", f"{ratio:.0%}")
    if not credit:
        st.info("A poor credit history is the strongest factor against approval.")
    if ratio > 0.35:
        st.info("The instalment is a high share of income - a smaller amount or longer term improves chances.")
    st.caption("Educational project: predictions come from a model trained on simulated data and are not real lending advice.")
