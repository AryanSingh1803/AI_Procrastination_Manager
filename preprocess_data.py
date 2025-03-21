import pandas as pd

# Load dataset
df = pd.read_csv("smartphone_usage.csv")

# Handle missing values (fill with mean)
df.fillna(df.mean(numeric_only=True), inplace=True)


# Normalize numerical columns for better comparison
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = ['Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Social_Media_Usage_Hours', 
            'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save preprocessed data
df.to_csv("processed_data.csv", index=False)

print("✅ Data preprocessing completed. Saved as 'processed_data.csv'.")
