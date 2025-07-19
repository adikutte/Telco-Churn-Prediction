import pandas as pd

# Load dataset
df = pd.read_csv("TelcoCustomerChurn.csv")

# Show shape and first 5 rows
print("Shape:", df.shape)
print(df.head())

# Check for null values
print("\nMissing values:\n", df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")

# Plot churn count
plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df, hue='Churn', palette='Set2', legend=False)
plt.title('Churn Distribution')
plt.show()

# Gender vs Churn
plt.figure(figsize=(5, 4))
sns.countplot(x='gender', hue='Churn', data=df, palette='Set1')
plt.title('Churn by Gender')
plt.show()

# Contract type vs Churn
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set3')
plt.title('Churn by Contract Type')
plt.show()

df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Check if any rows became NaN
print("Missing TotalCharges:", df['TotalCharges'].isnull().sum())
df.dropna(inplace=True)


from sklearn.preprocessing import LabelEncoder
#Encode target column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
# Label Encode binary columns
le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode remaining categorical columns
df = pd.get_dummies(df)
print("Final shape after encoding:", df.shape)


from sklearn.model_selection import train_test_split
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Train on SMOTE-resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)
# Predict on actual test set
y_pred = model.predict(X_test)
# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(acc * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pickle
# Save feature names used during training
feature_names = X_train.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
# Save the trained model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
# ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()
# AUC Score
auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {round(auc, 2)}")
