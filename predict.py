# predict.py
import joblib
import pandas as pd

# Load saved model and label encoder dictionary
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")  # This will store separate encoders per column

# Define the feature order (must match training)
feature_names = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Function to take user input
def get_user_input():
    user_data = {}
    for feature in feature_names:
        val = input(f"Enter {feature}: ")
        # If feature is categorical, encode using stored encoder
        if feature in encoders:
            val = encoders[feature].transform([val])[0]
        else:
            val = float(val)
        user_data[feature] = val
    return user_data

# Get processed user input
user_input_dict = get_user_input()

# Convert to DataFrame (keeps feature names)
user_df = pd.DataFrame([user_input_dict])

# Predict loan approval
prediction = model.predict(user_df)

if prediction[0] == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Not Approved")
