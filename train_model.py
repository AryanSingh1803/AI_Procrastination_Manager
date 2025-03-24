import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)

# Enable interactive mode for displaying plots
plt.ion()

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

# Split dataset (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Base Models
xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
logreg = LogisticRegression(max_iter=200, random_state=42)

# Hybrid Model: Voting Classifier (Soft Voting for better probabilities)
model = VotingClassifier(estimators=[('xgb', xgb), ('logreg', logreg)], voting='soft')

# Train the hybrid model
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
joblib.dump(model, "procrastination_hybrid_model.pkl")
print("✅ Hybrid model saved successfully!")

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# --- ROC Curve ---
y_test_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# Ensure all plots are displayed
plt.show(block=True)
