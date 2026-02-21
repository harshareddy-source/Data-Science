import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

#Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes (1).csv")
    return data

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    #Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train Model
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, X

#UI
st.title("Diabetes Prediction App 🩺")
st.write("Enter patient details below:")

# Load model and data
model, scaler, X = train_model()

inputs = []
for col in X.columns:
    val = st.number_input(col, value=float(X[col].mean()))
    inputs.append(val)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1,-1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Result: Patient is likely Diabetic (Risk: {prediction_proba:.1%})")
    else:
        st.success(f"Result: Patient is likely NOT Diabetic (Risk: {prediction_proba:.1%})")

# Add model info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("• Algorithm: Logistic Regression")
st.sidebar.write("• Dataset: Pima Indians Diabetes")
st.sidebar.write("• Features: 8 health metrics")
st.sidebar.write("• Accuracy: ~71%")
