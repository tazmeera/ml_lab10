"""
Lab Sheet-10 | Project 2: Disease Prediction System - training pipeline
MCA III Semester (Session 2026-2027)

Compares 7 classification algorithms (5-fold CV + hold-out test), reports accuracy / macro-F1 /
top-3 accuracy, per-disease metrics and confusion matrix, saves the best pipeline (disease_model.joblib).
Run: python generate_disease_dataset.py ; python train_disease_model.py
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
sns.set_theme(style="whitegrid")

df = pd.read_csv(os.path.join(HERE, "disease_data.csv"))
SYMPTOMS = [c for c in df.columns if c not in ("age", "gender", "body_temp_c", "disease")]
print(f"Shape {df.shape} | {len(SYMPTOMS)} symptoms | {df.disease.nunique()} diseases")
print("Missing:", df.isna().sum()[df.isna().sum() > 0].to_dict())

# ------------------------------------------------------------------ EDA
fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.7]})
df.disease.value_counts().sort_values().plot.barh(ax=ax[0], color="#4c78a8")
ax[0].set_title("Records per disease (balanced classes)")
prof = df.groupby("disease")[SYMPTOMS].mean()
sns.heatmap(prof, cmap="YlOrRd", ax=ax[1], cbar_kws={"label": "share of patients with symptom"})
ax[1].set_title("Symptom profile of each disease"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "eda_symptom_profiles.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ preprocessing + models
X = df.drop(columns="disease")
y = df.disease
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
NUM = ["age", "body_temp_c"]
pre = lambda: ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
    ("sym", "passthrough", SYMPTOMS),
])
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=SEED),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Naive Bayes": BernoulliNB(),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def top_k(pipe, X_, y_, k=3):
    proba = pipe.predict_proba(X_)
    top = pipe.classes_[np.argsort(proba, axis=1)[:, -k:]]
    return np.mean([t in row for t, row in zip(y_, top)])


rows, fitted = [], {}
for name, m in MODELS.items():
    pipe = Pipeline([("pre", pre()), ("model", m)])
    cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    rows.append(dict(Model=name, CV_Accuracy=cv_acc.mean(), CV_Std=cv_acc.std(), Test_Accuracy=accuracy_score(y_te, pred),
                     Macro_F1=f1_score(y_te, pred, average="macro"), Top3_Accuracy=top_k(pipe, X_te, y_te)))
    fitted[name] = pipe
    print(f"{name:20s} cv={cv_acc.mean():.3f}+-{cv_acc.std():.3f} test={rows[-1]['Test_Accuracy']:.3f} "
          f"f1={rows[-1]['Macro_F1']:.3f} top3={rows[-1]['Top3_Accuracy']:.3f}")
res = pd.DataFrame(rows).sort_values("CV_Accuracy", ascending=False).reset_index(drop=True)
best_name = res.loc[0, "Model"]

# ------------------------------------------------------------------ light tuning of the winner
grids = {"Logistic Regression": {"model__C": [0.1, 1, 10]}, "SVM (RBF)": {"model__C": [1, 3, 10]},
         "Random Forest": {"model__max_depth": [8, 14, None], "model__min_samples_leaf": [1, 3]},
         "Gradient Boosting": {"model__learning_rate": [0.05, 0.1], "model__n_estimators": [120, 200]},
         "KNN": {"model__n_neighbors": [7, 15, 25]}, "Decision Tree": {"model__max_depth": [6, 10, 14]},
         "Naive Bayes": {"model__alpha": [0.1, 1.0]}}
gs = GridSearchCV(Pipeline([("pre", pre()), ("model", MODELS[best_name])]), grids[best_name], cv=cv,
                  scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
final = gs.best_estimator_
pred = final.predict(X_te)
final_acc, final_f1, final_top3 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"), top_k(final, X_te, y_te)
print(f"\nBest: {best_name} | params {gs.best_params_} | test acc {final_acc:.3f} | macro-F1 {final_f1:.3f} | top-3 {final_top3:.3f}")
res = pd.concat([pd.DataFrame([dict(Model=f"{best_name} (tuned)", CV_Accuracy=gs.best_score_, CV_Std=np.nan,
                                    Test_Accuracy=final_acc, Macro_F1=final_f1, Top3_Accuracy=final_top3)]), res], ignore_index=True)
res.round(4).to_csv(os.path.join(HERE, "model_comparison.csv"), index=False)
report = classification_report(y_te, pred, digits=3)
open(os.path.join(HERE, "classification_report.txt"), "w").write(f"Model: {best_name} (tuned)\n\n{report}")
print(report)

# ------------------------------------------------------------------ evaluation plots
fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.5, 1]})
cm = confusion_matrix(y_te, pred, labels=final.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=final.classes_, yticklabels=final.classes_, ax=ax[0])
ax[0].set(title=f"Confusion matrix - {best_name} (tuned)", xlabel="Predicted", ylabel="Actual")
plt.setp(ax[0].get_xticklabels(), rotation=40, ha="right")
r = res.iloc[1:].melt(id_vars="Model", value_vars=["CV_Accuracy", "Test_Accuracy", "Top3_Accuracy"])
sns.barplot(data=r, y="Model", x="value", hue="variable", ax=ax[1]); ax[1].set_xlim(.5, 1.01)
ax[1].set_title("Algorithm comparison")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "model_evaluation.png"), dpi=150); plt.close()

# which symptoms matter? (permutation importance on the final model)
from sklearn.inspection import permutation_importance
pi = permutation_importance(final, X_te, y_te, n_repeats=8, random_state=SEED, scoring="accuracy", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(15)
plt.figure(figsize=(7, 5.5)); imp.plot.barh(color="#e45756")
plt.title("Most influential inputs (permutation importance)"); plt.tight_layout()
plt.savefig(os.path.join(HERE, "feature_importance.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ save
joblib.dump(final, os.path.join(HERE, "disease_model.joblib"))
json.dump({"best_model": best_name, "test_accuracy": round(final_acc, 4), "macro_f1": round(final_f1, 4),
           "top3_accuracy": round(final_top3, 4), "symptoms": SYMPTOMS, "classes": list(final.classes_),
           "top_features": list(imp.sort_values(ascending=False).head(5).index)},
          open(os.path.join(HERE, "model_metadata.json"), "w"), indent=2)
print("Saved disease_model.joblib + model_metadata.json")
