import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the models
model_v1 = load_model('HeartAttackRiskModel/heart_attack_model_v1.h5')
model_v2 = load_model('HeartAttackRiskModel/heart_attack_model_v2.h5')

# Function to get user input
def get_user_input():
    print("Please enter the following details for heart attack risk prediction:")
    age = float(input("Age: "))
    sex = int(input("Sex (1 = Male, 0 = Female): "))
    cp = int(input("Chest Pain Type (0-3): "))
    trestbps = float(input("Resting Blood Pressure (in mm Hg): "))
    chol = float(input("Serum Cholesterol (in mg/dL): "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dL (1 = True, 0 = False): "))
    restecg = int(input("Resting Electrocardiographic Results (0-2): "))
    thalach = float(input("Maximum Heart Rate Achieved: "))
    exang = int(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = int(input("Slope of Peak Exercise ST Segment (0-2): "))
    ca = int(input("Number of Major Vessels (0-4): "))
    thal = int(input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

# Get user input and prepare data
user_data = get_user_input()
input_df = pd.DataFrame([user_data])

# Standardize input based on training scaling
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_df)  # Note: Use training scaler in production

# Make predictions with both models
prediction_v1 = model_v1.predict(input_scaled)[0][0]
prediction_v2 = model_v2.predict(input_scaled)[0][0]

# Calculate average and overestimated risk
average_prediction = (prediction_v1 + prediction_v2) / 2
overestimated_risk = max(prediction_v1, prediction_v2)  # Use the higher risk for a cautious approach

# Output results
risk_average = "High" if average_prediction > 0.5 else "Low"
risk_overestimated = "High" if overestimated_risk > 0.5 else "Low"

print(f"Average Predicted Heart Attack Risk: {risk_average} (Probability: {average_prediction:.2f})")
print(f"Overestimated Heart Attack Risk: {risk_overestimated} (Probability: {overestimated_risk:.2f})")
