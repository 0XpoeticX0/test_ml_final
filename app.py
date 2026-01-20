import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("Diabetes_pre.pkl", "rb") as f:
    model = pickle.load(f)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], 
    columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        return "⚠ POSITIVE: High likelihood of Diabetes"
    else:
        return "✅ NEGATIVE: Low likelihood of Diabetes"

inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="Blood Pressure", value=70),
    gr.Number(label="Skin Thickness", value=20),
    gr.Number(label="Insulin", value=79),
    gr.Number(label="BMI", value=32.0),
    gr.Number(label="Diabetes Pedigree Function", value=0.5),
    gr.Number(label="Age", value=33)
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction Tool"
)

app.launch(share=True)