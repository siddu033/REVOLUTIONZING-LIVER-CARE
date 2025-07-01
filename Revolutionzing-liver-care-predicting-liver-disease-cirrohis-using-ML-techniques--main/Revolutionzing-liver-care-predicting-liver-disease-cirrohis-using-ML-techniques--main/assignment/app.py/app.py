import flask
from flask import Flask, request, render_template
import joblib,pickle
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('livercirrhosis1.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/inner-page')
def inner_page():
    return render_template('inner-page.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.form.to_dict()

        # Convert the input data to a DataFrame
        df = pd.DataFrame([data])
        
        # Ensure the columns are in the right order and include all necessary features
        expected_columns = [
            'Age', 'Duration of alcohol consumption(years)',
            'Quantity of alcohol consumption (quarters/day)', 'TCH', 'TG', 'LDL',
            'HDL', 'Hemoglobin  (g/dl)', 'PCV  (%)', 'MCV   (femtoliters/cell)',
            'Total Count', 'Polymorphs  (%) ', 'Lymphocytes  (%)',
            'Monocytes   (%)', 'Eosinophils   (%)', 'Basophils  (%)',
            'Platelet Count  (lakhs/mm)', 'Total Bilirubin    (mg/dl)',
            'Direct    (mg/dl)', 'Indirect     (mg/dl)', 'Total Protein     (g/dl)',
            'Albumin   (g/dl)', 'Globulin  (g/dl)', 'A/G Ratio',
            'AL.Phosphatase      (U/L)', 'SGOT/AST      (U/L)', 'SGPT/ALT (U/L)',
            'Systolic', 'Diastolic', 'Gender_female ', 'Gender_male',
            'Gender_transgender', 'Place(location where the patient lives)_rural',
            'Place(location where the patient lives)_urban',
            'Type of alcohol consumed_both',
            'Type of alcohol consumed_branded liquor',
            'Type of alcohol consumed_country liquor',
            'Hepatitis B infection_negative', 'Hepatitis B infection_positive',
            'Hepatitis C infection_negative', 'Hepatitis C infection_positive',
            'Obesity_yes', 'Family history of cirrhosis/ hereditary_no',
            'Family history of cirrhosis/ hereditary_yes',
            'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)_no'
        ]

        # Add any missing columns with default values of 0
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match the expected order
        df = df[expected_columns]

        # Make a prediction
        prediction = model.predict(df)
        
        # Convert all columns to numeric values
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill any NaN values with 0 (if any)
        df = df.fillna(0)

        # Make a prediction
        prediction = model.predict(df)
        
        # Map the prediction to 'yes' or 'no'
        prediction_label = 'yes' if prediction[0] == 1 else 'no'
        
        # Return the result to the template
        return render_template('result.html', prediction=prediction_label)
    
    except Exception as e:
        return render_template('error.html', error=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)