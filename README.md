# 🏦 Loan Approval Prediction

A Machine Learning-based web application that predicts loan approval based on applicant information. The application uses a trained classification model and provides predictions through a simple Flask web interface.

## 📌 Project Overview

Loan approval decisions depend on several applicant and loan-related factors such as income, education, credit history, employment status, and property area.

This project uses Machine Learning to analyze these factors and predict whether a loan application is likely to be approved.

The trained model is integrated into a Flask web application where users can enter applicant details and receive a prediction.

## 🎯 Problem Statement

To develop a web-based Machine Learning system that can predict loan approval based on applicant and loan-related information.

## ✨ Features

- Applicant information input
- Loan approval prediction
- Machine Learning model integration
- Flask-based web application
- Data preprocessing
- Prediction through a web interface
- Simple and user-friendly interface

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS
- **Model:** Random Forest Classifier
- **Model Storage:** Pickle

## 🤖 Machine Learning

The project uses a **Random Forest Classifier** for loan approval prediction.

The model was trained using applicant and loan-related features and saved as a `.pkl` file. The trained model is loaded by the Flask application to generate predictions for new applicant information.

## 📊 Input Features

The prediction system uses applicant and loan-related information such as:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

## 📂 Project Structure

```text
loan-approval-prediction/
│
├── app.py
├── model.pkl
├── data.csv
├── train_model.py
├── requirements.txt
│
├── scripts/
│   └── setup_and_train.py
│
├── templates/
│   └── index.html
│
├── .gitignore
└── .vscode/
    └── settings.json
