from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.predict import predict_pipeline

import os 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2 levels up from app.py
file_path = os.path.join(base_dir, 'data', 'default of credit card clients.xls')

df = pd.read_excel(file_path,header = 1)



app = FastAPI()

class CustomerRequest(BaseModel):
    cust_id: int

@app.get("/predict/")
async def predict(cust_id: int):
    # Here load your dataframe and call predict_pipeline for the given cust_id
    try:
        # For example, assume df is your full data
        cust_row = df[df['ID'] == cust_id]
        if cust_row.empty:
            raise HTTPException(status_code=404, detail=f"Customer ID {cust_id} not found")
        
        results = predict_pipeline(cust_row)
        result_columns = ['ID', 'LIMIT_BAL', 'PD', 'Predicted_Label', 'adjusted_limit', 'limit_action']
        return results[result_columns].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))