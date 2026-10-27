"""Shared feature engineering - imported by BOTH train_loan_model.py and app.py so the saved
joblib pipeline can be unpickled anywhere (a function defined in __main__ cannot)."""

NUM = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History",
       "Total_Income", "EMI_to_Income"]
CAT = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def add_features(X):
    """Total household income and EMI burden ratio (monthly instalment / monthly income)."""
    X = X.copy()
    X["Total_Income"] = X.ApplicantIncome + X.CoapplicantIncome
    X["EMI_to_Income"] = (X.LoanAmount * 1000 / X.Loan_Amount_Term) / X.Total_Income
    return X
