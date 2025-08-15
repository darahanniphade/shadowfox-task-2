import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")

# Drop Loan_ID because it's not a useful feature
data = data.drop("Loan_ID", axis=1)

# List of categorical columns (excluding target)
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Dictionary to store encoders
encoders = {}

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
data['Loan_Status'] = target_encoder.fit_transform(data['Loan_Status'])

# Split features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "loan_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("Model and encoders saved successfully.")
