import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final data
df = pd.read_csv("final_data.csv")

# Histogram of Procrastination Score
plt.figure(figsize=(8,5))
sns.histplot(df["Procrastination_Score"], bins=30, kde=True)
plt.title("Procrastination Score Distribution")
plt.xlabel("Procrastination Score")
plt.ylabel("Frequency")
plt.show()
