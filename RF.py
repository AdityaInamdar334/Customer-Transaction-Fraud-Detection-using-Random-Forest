# Re-import necessary libraries after execution state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Simulating a dataset for Customer Transaction Fraud Detection
np.random.seed(42)

num_samples = 5000
data = {
    'Transaction_Amount': np.random.uniform(10, 1000, num_samples),
    'Transaction_Time': np.random.randint(0, 24, num_samples),  # 24-hour format
    'Location_ID': np.random.randint(1, 50, num_samples),
    'Merchant_ID': np.random.randint(1, 100, num_samples),
    'Customer_Age': np.random.randint(18, 70, num_samples),
    'Num_Prev_Transactions': np.random.randint(0, 100, num_samples),
    'Card_Type': np.random.choice(['Debit', 'Credit', 'Prepaid'], num_samples),
    'Is_Fraud': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])  # 5% fraud cases
}

df = pd.DataFrame(data)

# Encoding categorical variables
df['Card_Type'] = df['Card_Type'].astype('category').cat.codes  # Convert to numerical values

# Splitting the dataset into training and testing sets
X = df.drop(columns=['Is_Fraud'])
y = df['Is_Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest model on balanced data
model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_resampled, y_resampled)

# Make predictions
y_pred_smote = model_smote.predict(X_test)

# Evaluate the improved model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
report_smote = classification_report(y_test, y_pred_smote)
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

# Display results
print(f"Model Accuracy After SMOTE: {accuracy_smote:.4f}")
print("\nClassification Report After SMOTE:\n", report_smote)

# Plot confusion matrix after SMOTE
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix After SMOTE")
plt.show()
