"""Draws flowchart.png and system_architecture.png for both capstone projects (Lab 10 guideline:
'Include flowcharts and system architecture'). Run: python make_diagrams.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
C = dict(data="#dbe9f6", proc="#e3f2dc", model="#fde9d0", ui="#efe0f5", out="#f9dcdc", edge="#34495e")


def box(ax, x, y, w, h, text, color, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=C["edge"], lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.5, color=C["edge"]))
    if text:
        ax.text((p[0] + q[0]) / 2 + .1, (p[1] + q[1]) / 2 + .12, text, fontsize=8, color="#555", ha="center")


def canvas(title, w=13, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(w / 2, h - .3, title, ha="center", fontsize=15, fontweight="bold")
    return fig, ax


def flowchart(path, title, steps, decision):
    """Two-row snake flowchart: steps = list of (text, colour)."""
    fig, ax = canvas(title)
    n = len(steps)
    top = steps[: (n + 1) // 2]
    bottom = steps[(n + 1) // 2:]
    xs = lambda k: [1.6 + i * (10.0 / max(k - 1, 1)) for i in range(k)]
    pos = [(x, 4.4) for x in xs(len(top))] + [(x, 1.6) for x in reversed(xs(len(bottom)))]
    for (x, y), (t, col) in zip(pos, steps):
        box(ax, x, y, 2.05, 1.05, t, col, 9.5)
    for i in range(n - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            d = 1.05 if x2 > x1 else -1.05
            arrow(ax, (x1 + d, y1), (x2 - d, y2))
        else:
            arrow(ax, (x1, y1 - .55), (x2, y2 + .55))
    ax.text(6.5, .35, decision, ha="center", fontsize=9.5, style="italic", color="#444")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def architecture(path, title, layers, side_note):
    """Layered architecture: layers = [(layer name, [component, ...], colour)] drawn top to bottom."""
    fig, ax = canvas(title, 13, 7)
    y = 5.6
    ys = []
    for name, comps, col in layers:
        ax.add_patch(FancyBboxPatch((.3, y - .75), 12.4, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white", ec="#bbb", lw=1, ls="--"))
        ax.text(.5, y + .55, name, fontsize=10, fontweight="bold", color="#555", va="center")
        k = len(comps)
        for i, t in enumerate(comps):
            box(ax, .3 + (12.4 / k) * (i + .5), y - .05, min(3.2, 12.4 / k - .5), .85, t, col, 9)
        ys.append(y)
        y -= 1.75
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, (6.5, a - .78), (6.5, b + .78))
    ax.text(6.5, .1, side_note, ha="center", fontsize=9.5, style="italic", color="#444")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Project 1: Loan approval
d1 = os.path.join(HERE, "p1_loan")
flowchart(f"{d1}/flowchart.png", "Loan Approval System - Workflow", [
    ("Load applicant\ndataset (CSV)", C["data"]), ("Explore data\n(EDA, charts)", C["data"]),
    ("Clean + impute\nmissing values", C["proc"]), ("Feature engineering\n(Total income, EMI ratio)", C["proc"]),
    ("Encode categories\n+ scale numbers", C["proc"]), ("Train 5 models\n(5-fold CV)", C["model"]),
    ("Tune best model\n(GridSearchCV)", C["model"]), ("Evaluate: accuracy,\nF1, ROC-AUC", C["model"]),
    ("Save pipeline\n(loan_model.joblib)", C["out"]), ("Streamlit form:\npredict approval", C["ui"]),
], "Same saved pipeline is used for training and prediction, so preprocessing is identical in the app.")
architecture(f"{d1}/system_architecture.png", "Loan Approval System - Architecture", [
    ("PRESENTATION LAYER", ["Streamlit web form\n(app.py)", "Result card:\nprobability + EMI ratio"], C["ui"]),
    ("MODEL / PIPELINE LAYER", ["Feature engineering\n(loan_features.py)", "ColumnTransformer:\nimpute, scale, one-hot", "Logistic Regression\n(tuned)"], C["model"]),
    ("DATA + STORAGE LAYER", ["loan_data.csv", "loan_model.joblib", "model_metadata.json"], C["data"]),
], "Train: train_loan_model.py writes the model files | Serve: app.py loads them with joblib.")

# ------------------------------------------------------------------ Project 2: Disease prediction
d2 = os.path.join(HERE, "p2_disease")
flowchart(f"{d2}/flowchart.png", "Disease Prediction System - Workflow", [
    ("Load symptom\ndataset (CSV)", C["data"]), ("EDA: symptom\nprofiles per disease", C["data"]),
    ("Impute age / temp,\nencode gender", C["proc"]), ("Stratified\ntrain / test split", C["proc"]),
    ("Train 7 classifiers\n(5-fold CV)", C["model"]), ("Pick + tune best\nmodel", C["model"]),
    ("Evaluate: accuracy,\nmacro-F1, top-3", C["model"]), ("Save pipeline\n(disease_model.joblib)", C["out"]),
    ("Streamlit: select\nsymptoms", C["ui"]), ("Show top-3 diseases\n+ probabilities", C["ui"]),
], "Educational system on simulated data - not a medical device.")
architecture(f"{d2}/system_architecture.png", "Disease Prediction System - Architecture", [
    ("PRESENTATION LAYER", ["Symptom picker +\nage / gender / temp", "Top-3 prediction chart\n+ disclaimer"], C["ui"]),
    ("MODEL / PIPELINE LAYER", ["Preprocessing:\nimpute, scale, one-hot", "7 candidate classifiers\n(compared by CV)", "Best model:\nLogistic Regression"], C["model"]),
    ("DATA + STORAGE LAYER", ["disease_data.csv", "disease_model.joblib", "model_metadata.json"], C["data"]),
], "Train: train_disease_model.py writes the model files | Serve: app.py loads them with joblib.")
print("diagrams written")
