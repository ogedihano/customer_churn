# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import os

app = Flask("__name__")

# Load the original data
df_1 = pd.read_csv("first_telc.csv")

# Load model and expected columns
model = pickle.load(open("model.sav", "rb"))
# If you have saved the training columns, load them here
# expected_columns = pickle.load(open("training_columns.pkl", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Collect all form inputs
    input_data = {
        'SeniorCitizen': request.form['query1'],
        'MonthlyCharges': request.form['query2'],
        'TotalCharges': request.form['query3'],
        'gender': request.form['query4'],
        'Partner': request.form['query5'],
        'Dependents': request.form['query6'],
        'PhoneService': request.form['query7'],
        'MultipleLines': request.form['query8'],
        'InternetService': request.form['query9'],
        'OnlineSecurity': request.form['query10'],
        'OnlineBackup': request.form['query11'],
        'DeviceProtection': request.form['query12'],
        'TechSupport': request.form['query13'],
        'StreamingTV': request.form['query14'],
        'StreamingMovies': request.form['query15'],
        'Contract': request.form['query16'],
        'PaperlessBilling': request.form['query17'],
        'PaymentMethod': request.form['query18'],
        'tenure': request.form['query19']
    }
    
    # Create DataFrame from input
    new_df = pd.DataFrame([input_data])
    
    # Preprocess the new data exactly like training data
    # 1. Create tenure groups
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'].astype(int), 
                                   range(1, 80, 12), 
                                   right=False, 
                                   labels=labels)
    
    # 2. One-hot encode categorical features
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    
    # Get dummies for new data
    new_df_encoded = pd.get_dummies(new_df[categorical_cols])
    
    # IMPORTANT: Align columns with training data
    # Get the columns the model expects
    expected_columns = model.feature_names_in_
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0
    
    # Ensure columns are in correct order
    new_df_encoded = new_df_encoded[expected_columns]
    
    # Make prediction
    prediction = model.predict(new_df_encoded)
    probability = model.predict_proba(new_df_encoded)[:,1]
    
    if prediction[0] == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = f"Confidence: {probability[0]*100:.2f}%"
    else:
        o1 = "This customer is likely to continue!!"
        o2 = f"Confidence: {probability[0]*100:.2f}%"
        
    return render_template('home.html', 
                         output1=o1, 
                         output2=o2,
                         **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)