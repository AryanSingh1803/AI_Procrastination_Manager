import pandas as pd  # Import Pandas

# ✅ Load CSV file (Make sure it's named correctly!)
file_path = r"C:\Users\KIIT\Desktop\AI_Procrastination_Manager\smartphone_usage.csv"
df = pd.read_csv(file_path)

# ✅ Print first 5 rows to check data
print(df.head())

# ✅ Display column names
print("Columns in dataset:", df.columns)
