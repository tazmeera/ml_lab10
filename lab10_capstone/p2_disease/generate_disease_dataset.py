"""Generates disease_data.csv (simulated patient symptom records) for Lab 10 Project 2.
Each disease has a probability profile over 26 symptoms (many symptoms overlap between diseases, as in
real life), plus age, gender and body temperature. EDUCATIONAL DATA ONLY - not medical knowledge."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
SYMPTOMS = ["fever", "high_fever", "chills", "cough", "dry_cough", "sore_throat", "runny_nose", "sneezing",
            "shortness_of_breath", "chest_pain", "fatigue", "body_ache", "headache", "nausea", "vomiting", "diarrhea",
            "abdominal_pain", "rash", "joint_pain", "loss_of_taste_smell", "sweating", "loss_of_appetite",
            "frequent_urination", "excessive_thirst", "blurred_vision", "light_sensitivity"]

# symptom probabilities per disease (unlisted symptoms get background noise 3-5 %)
PROFILE = {
    "Common Cold": dict(runny_nose=.9, sneezing=.85, sore_throat=.7, cough=.6, headache=.3, fatigue=.35, fever=.2),
    "Influenza": dict(fever=.9, high_fever=.4, chills=.7, body_ache=.85, headache=.7, fatigue=.8, cough=.75, dry_cough=.5, sore_throat=.5),
    "COVID-19": dict(fever=.75, dry_cough=.8, fatigue=.75, loss_of_taste_smell=.65, shortness_of_breath=.4, body_ache=.55, headache=.5, sore_throat=.4),
    "Dengue": dict(high_fever=.9, fever=.9, headache=.8, joint_pain=.8, body_ache=.8, rash=.55, nausea=.6, vomiting=.4, fatigue=.7, loss_of_appetite=.5),
    "Malaria": dict(high_fever=.8, fever=.85, chills=.9, sweating=.85, headache=.7, nausea=.55, vomiting=.45, body_ache=.55, fatigue=.65),
    "Typhoid": dict(high_fever=.75, fever=.9, abdominal_pain=.7, headache=.6, loss_of_appetite=.75, diarrhea=.4, fatigue=.65, rash=.25, nausea=.4),
    "Pneumonia": dict(fever=.8, cough=.9, shortness_of_breath=.75, chest_pain=.65, chills=.55, fatigue=.65, sweating=.4, loss_of_appetite=.4),
    "Gastroenteritis": dict(diarrhea=.9, vomiting=.8, nausea=.85, abdominal_pain=.8, fever=.4, fatigue=.5, loss_of_appetite=.55),
    "Migraine": dict(headache=.98, light_sensitivity=.85, nausea=.65, vomiting=.3, blurred_vision=.3, fatigue=.3),
    "Type 2 Diabetes": dict(frequent_urination=.85, excessive_thirst=.85, blurred_vision=.5, fatigue=.6, loss_of_appetite=.1, headache=.15),
}
AGE = {"Common Cold": (28, 18), "Influenza": (32, 20), "COVID-19": (40, 18), "Dengue": (27, 14), "Malaria": (30, 17),
       "Typhoid": (24, 12), "Pneumonia": (52, 22), "Gastroenteritis": (26, 18), "Migraine": (33, 10), "Type 2 Diabetes": (55, 12)}

rows = []
for disease, prof in PROFILE.items():
    for _ in range(420):
        r = {s: int(rng.random() < prof.get(s, rng.uniform(.03, .06))) for s in SYMPTOMS}
        # a few patients present with atypical (missing) key symptoms -> harder, realistic problem
        if rng.random() < .12:
            for s in list(prof)[:2]:
                r[s] = 0
        if r["high_fever"]:
            r["fever"] = 1
        mu, sd = AGE[disease]
        r["age"] = int(np.clip(rng.normal(mu, sd), 1, 90))
        r["gender"] = rng.choice(["Male", "Female"])
        r["body_temp_c"] = round(rng.normal(39.6 if r["high_fever"] else 38.4 if r["fever"] else 36.9, .35), 1)
        r["disease"] = disease
        rows.append(r)
df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
for col, frac in [("body_temp_c", .03), ("age", .01)]:
    df.loc[rng.choice(len(df), int(len(df) * frac), replace=False), col] = np.nan
df.to_csv("disease_data.csv", index=False)
print(df.shape, "| classes:", df.disease.nunique())
