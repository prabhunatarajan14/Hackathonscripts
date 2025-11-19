import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from openai import AzureOpenAI

# ======================================================
# 🟦 1. AZURE OPENAI CONFIG  (PASTE YOUR KEY LOCALLY)
# ======================================================

AZURE_KEY = "9eYGaOSJvZqvbj5ITlMbUUpTNvWBBWWisrXAdSHKFk2spAIfj9jQJQQJ99BJACHYHv6XJ3w3AAABACOGhzb7"        # <––– YOU paste key here in your file ONLY
AZURE_ENDPOINT = "https://mobpochack25-openai-18.openai.azure.com/"
AZURE_MODEL = "mobpochack25-openai-18"

USE_AI = True

client = AzureOpenAI(
    api_key=AZURE_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

# ======================================================
# 🟦 2. AI EXPLANATION FUNCTION
# ======================================================

def ask_ai_reasoning(input_data, predicted_label, assigned_auditor):
    if not USE_AI:
        return "AI disabled."

    prompt = f"""
    You are helping with real-time store audit reassignment.

    Input factors:
    {input_data}

    ML model decision: {predicted_label}
    Assigned auditor: {assigned_auditor}

    Explain clearly why the decision makes sense.
    """

    try:
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": "Explain in simple terms, short and clear."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"


# ======================================================
# 🟦 3. TRAFFIC + WEATHER FETCHERS (Placeholders)
# ======================================================

def get_live_traffic():
    # Normally you'd call Google API here
    # return 1 means heavy traffic, 0 means normal
    return np.random.randint(0, 2)

def get_live_weather():
    # Normally you'd call OpenWeather here
    # return 1 means bad weather, 0 means clear
    return np.random.randint(0, 2)


# ======================================================
# 🟦 4. TRAIN SAMPLE ML MODEL
# ======================================================

n_samples = 600
data = {
    'auditor_distance': np.random.randint(1, 60, n_samples),
    'auditor_skill': np.random.randint(1, 5, n_samples),
    'store_risk': np.random.randint(1, 10, n_samples),
    'audit_priority': np.random.randint(1, 5, n_samples),
    'store_open': np.random.randint(0, 2, n_samples),
    'traffic': np.random.randint(0, 2, n_samples),
    'weather': np.random.randint(0, 2, n_samples),
    'deadline_hrs': np.random.randint(1, 10, n_samples),
}

# Target logic
target = []
for i in range(n_samples):
    s = 0
    if data['auditor_distance'][i] > 30: s += 1
    if data['store_risk'][i] > 7: s += 1
    if data['audit_priority'][i] > 3: s += 1
    if data['traffic'][i] == 1: s += 1
    if data['weather'][i] == 1: s += 1
    if data['deadline_hrs'][i] < 4: s += 1
    if data['store_open'][i] == 0: s += 2
    target.append(1 if s >= 3 else 0)

df = pd.DataFrame(data)
df['reassign'] = target

X = df.drop('reassign', axis=1)
y = df['reassign']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

print("Model accuracy: ", accuracy_score(y_test, model.predict(X_test)))


# ======================================================
# 🟦 5. MULTI-AUDITOR ASSIGNMENT LOGIC
# ======================================================

auditors = [
    {"name": "A", "location": 5,  "skill": 3, "availability": 1},
    {"name": "B", "location": 12, "skill": 2, "availability": 1},
    {"name": "C", "location": 30, "skill": 4, "availability": 1},
]

def choose_best_auditor(row):
    """Pick the best auditor based on distance + skill + availability."""
    scores = []

    for a in auditors:
        if a["availability"] == 0:
            continue

        score = (
            (60 - abs(a["location"] - row['auditor_distance'])) +
            (a["skill"] * 10)
        )

        scores.append((score, a["name"]))

    if not scores:
        return None

    return max(scores)[1]  # return auditor with highest score


# ======================================================
# 🟦 6. USER INPUT
# ======================================================

print("\nEnter today’s store audit details:\n")

user_input = {
    "auditor_distance": int(input("Distance to store (km): ")),
    "auditor_skill": int(input("Skill needed (1-4): ")),
    "store_risk": int(input("Store risk level (1-10): ")),
    "audit_priority": int(input("Audit priority (1-4): ")),
    "store_open": int(input("Store open? (1=yes, 0=no): ")),
    "deadline_hrs": int(input("Hours before deadline: ")),
}

# Fetch LIVE data
user_input["traffic"] = get_live_traffic()
user_input["weather"] = get_live_weather()

print("\nLive traffic =", user_input["traffic"])
print("Live weather =", user_input["weather"])

input_df = pd.DataFrame([user_input])

# ML prediction

input_df = input_df[X_train.columns]
prediction = model.predict(input_df)[0]
decision = "Reassign" if prediction == 1 else "No Reassign"

# Assign auditor
assigned_auditor = choose_best_auditor(user_input) if decision == "Reassign" else "Current auditor"

print("\n🔍 Final Decision:", decision)
print("👤 Assigned Auditor:", assigned_auditor)

# AI Explanation
print("\n🤖 AI Reasoning:")
print(ask_ai_reasoning(user_input, decision, assigned_auditor))
