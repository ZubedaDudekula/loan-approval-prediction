import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess the loan data"""
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Handle missing values
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    
    # Create new features
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
    data['Balance_Income'] = data['Total_Income'] - (data['EMI'] * 1000)
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    
    # Handle Dependents column (convert to numeric)
    data['Dependents'] = data['Dependents'].replace('3+', '3')
    data['Dependents'] = pd.to_numeric(data['Dependents'])
    
    # Encode target variable
    le_target = LabelEncoder()
    data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
    le_dict['Loan_Status'] = le_target
    
    return data, le_dict

def train_model():
    """Train the loan approval model"""
    print("Loading data...")
    df = pd.read_csv('data.csv')
    
    print("Preprocessing data...")
    data, le_dict = preprocess_data(df)
    
    # Select features
    feature_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History', 'Gender', 'Married', 
                   'Dependents', 'Education', 'Self_Employed', 'Property_Area',
                   'Total_Income', 'EMI', 'Balance_Income']
    
    X = data[feature_cols]
    y = data['Loan_Status']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and encoders
    model_data = {
        'model': model,
        'encoders': le_dict,
        'feature_columns': feature_cols
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'model.pkl'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

if __name__ == "__main__":
    train_model()
