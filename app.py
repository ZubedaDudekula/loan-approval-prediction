from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
encoders = model_data['encoders']
feature_columns = model_data['feature_columns']

def preprocess_input(data):
    """Preprocess user input for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Calculate derived features
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        if col in encoders:
            try:
                df[col] = encoders[col].transform([df[col].iloc[0]])[0]
            except ValueError:
                # Handle unseen categories
                df[col] = 0
    
    # Handle Dependents
    if df['Dependents'].iloc[0] == '3+':
        df['Dependents'] = 3
    df['Dependents'] = pd.to_numeric(df['Dependents'])
    
    # Select only the features used in training
    df = df[feature_columns]
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': int(request.form['credit_history']),
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            'Self_Employed': request.form['self_employed'],
            'Property_Area': request.form['property_area']
        }
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Get prediction result
        if prediction == 1:
            result = "Approved"
            confidence = probability[1] * 100
        else:
            result = "Rejected"
            confidence = probability[0] * 100
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.1f}%",
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        return jsonify({
            'loan_status': 'Approved' if prediction == 1 else 'Rejected',
            'probability': {
                'approved': float(probability[1]),
                'rejected': float(probability[0])
            },
            'confidence': float(max(probability))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
