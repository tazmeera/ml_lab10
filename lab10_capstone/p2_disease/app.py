"""Disease Prediction System - Streamlit UI.   Run:  streamlit run app.py"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def load():
    model = joblib.load(os.path.join(HERE, "disease_model.joblib"))
    meta = json.load(open(os.path.join(HERE, "model_metadata.json")))
    return model, meta


model, meta = load()
SYMPTOMS = meta["symptoms"]
pretty = lambda s: s.replace("_", " ").capitalize()

st.title("🩺 Disease Prediction System")
st.warning("Educational project trained on SIMULATED data. It is NOT a medical device and must not be used for "
           "diagnosis. See a doctor for any health concern.")

with st.form("patient"):
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 100, 30)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    temp = c3.number_input("Body temperature (°C)", 35.0, 42.0, 37.0, 0.1)
    chosen = st.multiselect("Select the symptoms the patient has", SYMPTOMS, format_func=pretty)
    go = st.form_submit_button("Predict", type="primary")

if go:
    if not chosen:
        st.error("Please select at least one symptom.")
    else:
        row = {s: int(s in chosen) for s in SYMPTOMS}
        if "high_fever" in chosen:
            row["fever"] = 1
        row.update(age=age, gender=gender, body_temp_c=temp)
        proba = pd.Series(model.predict_proba(pd.DataFrame([row]))[0], index=model.classes_).sort_values(ascending=False)
        top = proba.head(3)
        st.subheader(f"Most likely: {top.index[0]}  ({top.iloc[0]:.0%})")
        st.bar_chart(top.rename("probability"))
        for name, p in top.items():
            st.write(f"**{name}** - {p:.1%}")
        if top.iloc[0] < 0.5:
            st.info("The model is not confident - symptoms overlap between several conditions. A medical test is needed.")
        st.caption(f"Model: {meta['best_model']} | test accuracy {meta['test_accuracy']:.1%} | top-3 accuracy {meta['top3_accuracy']:.1%}")
