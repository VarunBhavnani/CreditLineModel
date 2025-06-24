import pandas as pd
import joblib
import json

from model_utility import apply_woe_binning, adjust_credit_limit  # Your custom functions

# Load model
model = joblib.load('models/pd_model.pkl')

# Load woe_temp from JSON
with open('models/woe_bins.json', 'r') as f:
    woe_temp = pd.read_json(f, orient='records', lines=True)



def predict_pd(input_df):
    # Preprocess using WOE
    transformed_df = apply_woe_binning(input_df.copy(), woe_config)
    
    # Drop any rows with NaNs post-WOE
    X = transformed_df.drop(columns=['ID'], errors='ignore').dropna()

    # Predict
    pd_probs = model.predict_proba(X)[:, 1]
    input_df['pred_pd'] = pd_probs

    # Credit limit logic
    input_df['adjusted_limit'] = input_df.apply(
        lambda x: adjust_credit_limit(x['LIMIT_BAL'], x['pred_pd']), axis=1
    )

    return input_df[['ID', 'LIMIT_BAL', 'pred_pd', 'adjusted_limit']]
