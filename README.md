# 🏦 Credit Line Recommendation System

A complete end-to-end project that predicts **Probability of Default (PD)** and recommends an **adjusted credit line** for credit card customers.

Built using:
- ✅ FastAPI (Backend API)
- ✅ Streamlit (Frontend UI)
- ✅ Scikit-learn / Statsmodels (Modeling)
- ✅ WOE Binning & Logistic Regression
- ✅ Joblib, Pandas, Numpy for preprocessing

---

## 📁 Project Structure
CreditLineModel/
│
├── app/
│ ├── app.py # FastAPI app
│ └── app_ui.py # Streamlit UI
│
├── src/
│ ├── predict.py # Model pipeline logic
│ └── model_utility.py # WOE transforms, preprocessing
│
├── models/
│ ├── logit_model.pkl # Trained logistic regression model
│ └── woe_bins.json # WOE transformation config
│
├── data/
│ └── default of credit card clients.xls # Input dataset
│
├── requirements.txt
└── README.md

## ⚙️ Setup Instructions

### 1. Clone and navigate to the project
```bash
git clone https://github.com/VarunBhavnani/CreditLineModel
cd "CreditLineModel"
```

### 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

## Running the Project 

### 1. Start FastAPI Backend
- uvicorn app.app:app --reload 
Once running, test at:
http://127.0.0.1:8000/docs ✅

### 2. Launch Streamlit UI (in a new terminal)
streamlit run app/app_ui.py

### Usage Instruction
Go to the Streamlit app in your browser.
Enter a Customer ID (e.g., 22).
Click "Get Prediction".
View results including:
    LIMIT_BAL
    PD (Predicted Probability of Default)
    Predicted_Label (Good/Bad)
    adjusted_limit (Recommended Credit Limit)
    limit_action (Increase / Decrease / Keep same)


## Dependencies
pandas
numpy
scikit-learn
statsmodels
fastapi
uvicorn
joblib
openpyxl
xlrd
streamlit


## Appendix
### Data Set source: 
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

### Dataset Information

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

### Column Info

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
- X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
- X2: Gender (1 = male; 2 = female).
- X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
- X4: Marital status (1 = married; 2 = single; 3 = others).
- X5: Age (year).
- X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
- X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
- X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.