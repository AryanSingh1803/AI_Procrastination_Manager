import pandas as pd
import joblib

# Load model
model = joblib.load("procrastination_model.pkl")

# Sample input for prediction
sample = pd.DataFrame({
    "Total_App_Usage_Hours": [0.7],
    "Daily_Screen_Time_Hours": [0.6],
    "Social_Media_Usage_Hours": [0.9],
    "Productivity_App_Usage_Hours": [0.3],
    "Gaming_App_Usage_Hours": [0.8]
})

# Predict
prediction = model.predict(sample)
print("🟢 Procrastination Detected" if prediction[0] == 1 else "🔵 Low Procrastination")
