"""
Lab Sheet-10 | Project 1: Loan Approval Prediction System - training pipeline
MCA III Semester (Session 2026-2027)

Steps: EDA -> preprocessing (impute, encode, scale) -> compare 6 algorithms with 5-fold CV ->
       evaluate on held-out test set -> tune the best -> save pipeline (loan_model.joblib).
Everything (imputation + encoding + model) lives in ONE sklearn Pipeline so the Streamlit app
applies exactly the same preprocessing as training (no train/serve mismatch).

Run: python generate_loan_dataset.py ; python train_loan_model.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
sns.set_theme(style="whitegrid")

df = pd.read_csv(os.path.join(HERE, "loan_data.csv"))
print(f"Shape {df.shape}\nMissing values:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}")
print("\nTarget balance:\n", df.Loan_Status.value_counts(normalize=True).round(3).to_string())

# ------------------------------------------------------------------ EDA charts
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
sns.countplot(data=df, x="Loan_Status", ax=ax[0, 0], hue="Loan_Status", legend=False)
ax[0, 0].set_title("Approval distribution")
sns.barplot(data=df, x="Credit_History", y=(df.Loan_Status == "Y").astype(int), ax=ax[0, 1], color="#4c78a8")
ax[0, 1].set(title="Approval rate by credit history", ylabel="P(approved)")
sns.barplot(data=df, x="Education", y=(df.Loan_Status == "Y").astype(int), ax=ax[0, 2], color="#54a24b")
ax[0, 2].set(title="Approval rate by education", ylabel="P(approved)")
sns.barplot(data=df, x="Property_Area", y=(df.Loan_Status == "Y").astype(int), ax=ax[1, 0], color="#f58518")
ax[1, 0].set(title="Approval rate by property area", ylabel="P(approved)")
sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome", ax=ax[1, 1], showfliers=False)
ax[1, 1].set_title("Applicant income vs outcome")
sns.boxplot(data=df, x="Loan_Status", y="LoanAmount", ax=ax[1, 2], showfliers=False)
ax[1, 2].set_title("Loan amount (Rs '000) vs outcome")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "eda_charts.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ preprocessing pipeline
from loan_features import CAT, NUM, add_features  # shared with app.py


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])


def make_pipe(model):
    return Pipeline([("fe", FunctionTransformer(add_features)), ("pre", make_pre()), ("model", model)])


X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = (df.Loan_Status == "Y").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, m in MODELS.items():
    pipe = make_pipe(m)
    cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    pipe.fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    rows.append(dict(Model=name, CV_Accuracy=cv_acc, Accuracy=accuracy_score(y_te, pred),
                     Precision=precision_score(y_te, pred), Recall=recall_score(y_te, pred),
                     F1=f1_score(y_te, pred), ROC_AUC=roc_auc_score(y_te, proba)))
    fitted[name] = pipe
    print(f"{name:20s} cv={cv_acc:.3f} test_acc={rows[-1]['Accuracy']:.3f} auc={rows[-1]['ROC_AUC']:.3f}")
res = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------ tune best tree-based / linear model
best_name = res.loc[0, "Model"]
grids = {
    "Random Forest": {"model__max_depth": [4, 6, 10, None], "model__min_samples_leaf": [1, 5, 10]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Decision Tree": {"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [5, 20]},
    "KNN": {"model__n_neighbors": [5, 15, 31]},
}
gs = GridSearchCV(make_pipe(MODELS[best_name]), grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
final = gs.best_estimator_
pred, proba = final.predict(X_te), final.predict_proba(X_te)[:, 1]
print(f"\nBest algorithm: {best_name} | tuned params: {gs.best_params_}")
final_row = dict(Model=f"{best_name} (tuned)",
                 CV_Accuracy=cross_val_score(final, X_tr, y_tr, cv=cv, scoring="accuracy").mean(), Accuracy=accuracy_score(y_te, pred),
                 Precision=precision_score(y_te, pred), Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred),
                 ROC_AUC=roc_auc_score(y_te, proba))
res = pd.concat([pd.DataFrame([final_row]), res], ignore_index=True)
res.round(4).to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)
print("\n" + res.round(3).to_string(index=False))
print("\n" + classification_report(y_te, pred, target_names=["Rejected", "Approved"]))

# ------------------------------------------------------------------ evaluation plots
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Rejected", "Approved"], cmap="Blues", ax=ax[0], colorbar=False)
ax[0].set_title(f"Confusion matrix - {best_name} (tuned)")
for name, p in fitted.items():
    RocCurveDisplay.from_estimator(p, X_te, y_te, ax=ax[1], name=name)
ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_title("ROC curves")
r = res.iloc[1:].melt(id_vars="Model", value_vars=["Accuracy", "F1", "ROC_AUC"])
sns.barplot(data=r, y="Model", x="value", hue="variable", ax=ax[2]); ax[2].set_xlim(.5, 1); ax[2].set_title("Algorithm comparison")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "model_evaluation.png"), dpi=150); plt.close()

# feature importance (permutation importance works for any model)
from sklearn.inspection import permutation_importance
pi = permutation_importance(final, X_te, y_te, n_repeats=15, random_state=SEED, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="#4c78a8")
plt.title("Permutation importance (drop in ROC-AUC)"); plt.tight_layout()
plt.savefig(os.path.join(HERE, "feature_importance.png"), dpi=150); plt.close()
print("\nTop drivers:", ", ".join(imp.sort_values(ascending=False).head(3).index))

# ------------------------------------------------------------------ save
joblib.dump(final, os.path.join(HERE, "loan_model.joblib"))
meta = {"best_model": best_name, "params": {k: (v if v is not None else "None") for k, v in gs.best_params_.items()},
        "test_accuracy": round(final_row["Accuracy"], 4),
        "recall_approved": round(final_row["Recall"], 4), "recall_rejected": round(recall_score(y_te, pred, pos_label=0), 4), "test_roc_auc": round(final_row["ROC_AUC"], 4),
        "top_features": list(imp.sort_values(ascending=False).head(3).index)}
json.dump(meta, open(os.path.join(HERE, "model_metadata.json"), "w"), indent=2)
print("Saved loan_model.joblib + model_metadata.json")
