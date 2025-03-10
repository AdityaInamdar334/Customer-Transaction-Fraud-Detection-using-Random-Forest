import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Initializing and training the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Displaying results
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
