import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the preprocessed training model
model = joblib.load("procrastination_model.pkl")

# Load the raw test dataset (before preprocessing)
df_test = pd.read_csv("testing_data.csv")

# Load the processed training data (preprocessed before) to use for filling missing columns and fitting the scaler
df_train = pd.read_csv("processed_data.csv")

# Check for missing columns and fill them with the mean of the training dataset (if necessary)
required_columns = ['Total_App_Usage_Hours', 'Daily_Screen_Time_Hours', 'Social_Media_Usage_Hours', 
                    'Productivity_App_Usage_Hours', 'Gaming_App_Usage_Hours']

missing_columns = [col for col in required_columns if col not in df_test.columns]
if missing_columns:
    print(f"⚠️ Missing columns in the test dataset: {missing_columns}")
    for col in missing_columns:
        # Fill missing columns with mean value from training data (can change based on your use case)
        df_test[col] = df_train[col].mean()

# Handle missing values in the test dataset (fill with mean) - same as in training preprocessing
df_test.fillna(df_test.mean(numeric_only=True), inplace=True)

# Normalize numerical columns in the test dataset using the same scaler from the training preprocessing
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(df_train[required_columns])  # Fit the scaler on the training data

# Normalize the test data using the fitted scaler
df_test[required_columns] = scaler.transform(df_test[required_columns])

# Define the features for prediction
X_test = df_test[required_columns]

# Predict on the test dataset
y_test_pred = model.predict(X_test)

# Print predictions
print(f"Predictions: {y_test_pred}")

# If actual labels are available in the test dataset, you can compute performance metrics
if 'Procrastination_Label' in df_test.columns:
    y_test = df_test['Procrastination_Label']
    
    # Evaluate model performance
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"✅ Test Accuracy: {accuracy_test:.2f}")
    print(f"🎯 Precision: {precision:.2f}")
    print(f"🎯 Recall: {recall:.2f}")
    print(f"🎯 F1 Score: {f1:.2f}")
else:
    print("⚠️ No actual labels in the test dataset to evaluate performance.")
