import pandas as pd

# Load preprocessed data
df = pd.read_csv("processed_data.csv")

# Define a Procrastination Score
df["Procrastination_Score"] = (df["Social_Media_Usage_Hours"] * 0.6) + (df["Gaming_App_Usage_Hours"] * 0.4)

# Save the updated data
df.to_csv("final_data.csv", index=False)

print("✅ Feature engineering completed. Saved as 'final_data.csv'.")
