"""Generates loan_data.csv (simulated applicant data, Loan Prediction style) for Lab 10 Project 1.
Approval depends mostly on credit history, then income vs loan amount (EMI burden), education,
property area. Missing values are injected on purpose to practise preprocessing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
N = 3000
gender = rng.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.17, 0.09])
education = rng.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_emp = rng.choice(["Yes", "No"], N, p=[0.14, 0.86])
area = rng.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])
income = np.round(rng.lognormal(8.35, 0.55, N) * np.where(education == "Graduate", 1.15, 0.9), -1).clip(1500, 80000)
co_income = np.where(married == "Yes", np.round(rng.lognormal(7.2, 0.9, N), -1), 0).clip(0, 40000)
co_income = np.where(rng.random(N) < 0.3, 0, co_income)
loan = np.round((income + co_income) / 1000 * rng.uniform(8, 60, N)).clip(20, 700)  # Rs thousand (~8-60x monthly income)
term = rng.choice([360, 180, 480, 300, 240, 120, 84], N, p=[0.80, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01])
credit = (rng.random(N) < 0.84).astype(int)

emi_ratio = (loan * 1000 / term) / (income + co_income)
z = (-1.9 + 3.4 * credit - 14.0 * np.clip(emi_ratio - 0.06, 0, None) + 0.35 * (education == "Graduate")
     + 0.5 * (area == "Semiurban") - 0.2 * (area == "Rural") + 0.25 * (married == "Yes")
     - 0.2 * (self_emp == "Yes") + rng.normal(0, 0.6, N))
status = np.where(rng.random(N) < 1 / (1 + np.exp(-z)), "Y", "N")

df = pd.DataFrame({"Loan_ID": [f"LP{i:05d}" for i in range(1, N + 1)], "Gender": gender, "Married": married,
                   "Dependents": dependents, "Education": education, "Self_Employed": self_emp,
                   "ApplicantIncome": income.astype(int), "CoapplicantIncome": co_income.astype(int),
                   "LoanAmount": loan, "Loan_Amount_Term": term, "Credit_History": credit.astype(float),
                   "Property_Area": area, "Loan_Status": status})
for col, frac in [("Gender", .02), ("Married", .01), ("Dependents", .025), ("Self_Employed", .05),
                  ("LoanAmount", .04), ("Loan_Amount_Term", .02), ("Credit_History", .08)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan
df.to_csv("loan_data.csv", index=False)
print(f"loan_data.csv: {df.shape} | approval rate {np.mean(status == 'Y'):.1%}")
