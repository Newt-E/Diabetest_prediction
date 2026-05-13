import pickle 
import streamlit as st

diabates_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title("Diabetes Prediction Web App")

Pregnancies = st.number_input('Number of Pregnancies')
Glucose = st.number_input('Glucose Level')
BloodPressure = st.number_input('Blood Pressure value')
SkinThickness = st.number_input('Skin Thickness value')
Insulin = st.number_input('Insulin Level')
BMI = st.number_input('BMI value')
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
Age = st.number_input('Age of the Person')


diagnosis = ''

if st.button('Diabetes Test Result'):

    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    input_data_scaled = scaler.transform(input_data)
    diabates_prediction = diabates_model.predict(input_data_scaled)

    if (diabates_prediction[0] == 1):
        diagnosis = 'The person is diabetic'
    else:
        diagnosis = 'The person is not diabetic'

    st.success(diagnosis)