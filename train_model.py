import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("final_data.csv")

# Check if necessary columns are available in the dataset
required_columns = ['Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Social_Media_Usage_Hours', 
                    'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours', 'Procrastination_Score']

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"⚠️ Missing columns in dataset: {missing_columns}")
else:
    print("✅ All required columns are present.")

# Define labels (High Procrastination if Score > 0.5)
df["Procrastination_Label"] = df["Procrastination_Score"].apply(lambda x: 1 if x > 0.5 else 0)

# Features & Target
X = df[['Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Social_Media_Usage_Hours', 
       'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']]
y = df["Procrastination_Label"]

# Split dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on Train & Test Data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate Model
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"✅ Train Accuracy: {accuracy_train:.2f}")
print(f"✅ Test Accuracy: {accuracy_test:.2f}")
print(f"🎯 Precision: {precision:.2f}")
print(f"🎯 Recall: {recall:.2f}")
print(f"🎯 F1 Score: {f1:.2f}")

# Check Overfitting/Underfitting
if accuracy_train > accuracy_test + 0.1:
    print("⚠️ Possible Overfitting: Train accuracy is much higher than test accuracy!")
elif accuracy_test > accuracy_train + 0.1:
    print("⚠️ Possible Underfitting: Test accuracy is higher than train accuracy!")
else:
    print("✅ Model is well-generalized.")

# Save Model
joblib.dump(model, "procrastination_model.pkl")
print("✅ Model saved successfully!")
