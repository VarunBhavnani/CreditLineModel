import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import json
import joblib

import os
sys.path.append(os.path.abspath('../src'))

import model_utility as util

# Load model
model = joblib.load('../models/logit_model.pkl')

# Load woe_temp from JSON
with open('../models/woe_bins.json', 'r') as f:
    woe_temp = pd.read_json(f, orient='records', lines=True)

# Final model features used for prediction (excluding target)
final_model_features = [
    "Paid Jun", "PAY_AMT_Apr", "PAY_Jul", "LIMIT_BAL", "Paid Sep",
    "PAY_AMT_Jul", "utilization_ratio Apr", "PAY_AMT_Jun", "SEX_Male",
    "PAY_AMT_Aug", "MARRIAGE_Single", "PAY_Sep", "PAY_Apr"
]

months_info = [
    ('Sep', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1'),
    ('Aug', 'PAY_2', 'BILL_AMT2', 'PAY_AMT2'),
    ('Jul', 'PAY_3', 'BILL_AMT3', 'PAY_AMT3'),
    ('Jun', 'PAY_4', 'BILL_AMT4', 'PAY_AMT4'),
    ('May', 'PAY_5', 'BILL_AMT5', 'PAY_AMT5'),
    ('Apr', 'PAY_6', 'BILL_AMT6', 'PAY_AMT6'),
]

education_mapping = {
    1: 'Graduate School', 2: 'University', 3: 'High School',
    4: "Others", 5: "Others", 6: "Others"
}

gender_mapping = {1: 'Male', 2: 'Female'}

marital_mapping = {1: 'Married', 2: 'Single', 3: 'Others'}

# Use actual mode values from training
mode_education_train = 2  # 'University'
mode_marriage_train = 2   # 'Married'

def predict_pipeline(input_df):
    # Drop target column if present
    if 'default payment next month' in input_df.columns:
        input_df = input_df.drop(columns=['default payment next month'])

    df_temp = util.generate_monthly_features(input_df, months_info)

    # Rename columns for clarity
    df_temp.rename(columns={
        'PAY_0': 'PAY_Sep', 'PAY_2': 'PAY_Aug', 'PAY_3': 'PAY_Jul',
        'PAY_4': 'PAY_Jun', 'PAY_5': 'PAY_May', 'PAY_6': 'PAY_Apr',
        'BILL_AMT1': 'BILL_AMT_Sep', 'BILL_AMT2': 'BILL_AMT_Aug',
        'BILL_AMT3': 'BILL_AMT_Jul', 'BILL_AMT4': 'BILL_AMT_Jun',
        'BILL_AMT5': 'BILL_AMT_May', 'BILL_AMT6': 'BILL_AMT_Apr',
        'PAY_AMT1': 'PAY_AMT_Sep', 'PAY_AMT2': 'PAY_AMT_Aug',
        'PAY_AMT3': 'PAY_AMT_Jul', 'PAY_AMT4': 'PAY_AMT_Jun',
        'PAY_AMT5': 'PAY_AMT_May', 'PAY_AMT6': 'PAY_AMT_Apr'
    }, inplace=True)

    #df_temp = util.reorder_columns_monthwise(df_temp)

    # Clean and map EDUCATION, SEX, MARRIAGE
    df_temp['EDUCATION'] = np.where(df_temp['EDUCATION'] == 0, mode_education_train, df_temp['EDUCATION'])
    df_temp['EDUCATION'] = df_temp['EDUCATION'].map(education_mapping)

    df_temp['SEX'] = df_temp['SEX'].map(gender_mapping)

    df_temp['MARRIAGE'] = np.where(df_temp['MARRIAGE'] == 0, mode_marriage_train, df_temp['MARRIAGE'])
    df_temp['MARRIAGE'] = df_temp['MARRIAGE'].map(marital_mapping)

    # One-hot encoding
    df_temp = pd.get_dummies(df_temp, columns=['EDUCATION', 'SEX', 'MARRIAGE'], drop_first=False)

    # Drop unnecessary dummies
    col_to_drop = ['EDUCATION_Others', 'SEX_Female', 'MARRIAGE_Others']
    df_temp.drop(columns=[col for col in col_to_drop if col in df_temp.columns], inplace=True)
    
    # Ensure all final_model_features are present in df_temp
    for col in final_model_features:
        if col not in df_temp.columns:
            df_temp[col] = 0


    # Apply WOE transformation
    df_temp_woe = util.apply_woe_binning(df_temp, woe_temp)

    # Filter only model input features
    X_model = df_temp_woe[final_model_features]
    
    X_model = X_model.fillna(0)  # simple fix if it makes sense


    # Add constant (for statsmodels)
    X_model = sm.add_constant(X_model, has_constant='add')

    # Predict probability (PD)
    pd_prob = model.predict(X_model)

    # Predict binary label
    pd_label = (pd_prob >= 0.5).astype(int)

    # Append to original data
    df_temp['PD'] = pd_prob
    df_temp['Predicted_Label'] = pd_label

    # Adjust limit & get action
    df_temp['adjusted_limit'] = df_temp.apply(
        lambda x: util.adjust_credit_limit(x['LIMIT_BAL'], x['PD']), axis=1
    )
    df_temp['limit_action'] = df_temp['PD'].apply(util.get_limit_action)

    return df_temp