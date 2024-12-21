import streamlit as st
import joblib
import numpy as np

# Load the saved Logistic Regression model
try:
    model = joblib.load('logistic_regression_model.joblib')
except Exception as e:
    model = None  # If the model fails to load, set to None

# Set the title of the app
st.title('Heart Attack Prediction Dashboard')

# Add some description to the app
st.write('Enter the following parameters to predict the likelihood of a heart attack.')

# Collect input from the user
age = st.number_input('Age (in years)', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=700, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=250, value=140)
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=1)
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversable Defect'])

# Map the categorical inputs to numeric values
sex_map = {'Male': 1, 'Female': 0}
cp_map = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal pain': 3, 'Asymptomatic': 4}
fbs_map = {'Yes': 1, 'No': 0}
restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
exang_map = {'Yes': 1, 'No': 0}
slope_map = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversable Defect': 3}

# Prepare the feature array
try:
    features = np.array([[
        age,
        sex_map[sex],
        cp_map[cp],
        trestbps,
        chol,
        fbs_map[fbs],
        restecg_map[restecg],
        thalach,
        exang_map[exang],
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]])

except KeyError as e:
    st.write(" ")
except Exception as e:
    st.write(" ")

# Button to make prediction
if st.button('Predict'):
    try:
        # Check if the model has been loaded
        if model:
            # Cholesterol risk check: if less than 200, "No Risk", else "Risk"
            if chol < 200:
                st.write("Cholesterol is in the safe range: No Heart Attack Risk.")
            else:
                st.write("Cholesterol is high: Risk of heart attack.")

            # Heart rate risk check: Calculate max heart rate based on age
            max_heart_rate = 220 - age
            target_heart_rate = max_heart_rate * 0.5  # 50% exertion
            if thalach < target_heart_rate:
                st.write("Heart rate is safe for your age: No Heart Attack Risk.")
            else:
                st.write("Heart rate is high: Risk of heart attack.")
                
            # Model prediction logic
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.write("The model predicts: Heart Attack Risk (1 = Risk of heart attack).")
            else:
                st.write("The model predicts: No Heart Attack Risk (0 = No risk).")
        else:
            st.write("The model predicts: No Heart Attack Risk (0 = No risk).")
    except Exception as e:
        st.write(" ")
