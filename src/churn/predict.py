import joblib
import pandas as pd
from src.churn.preprocessing import preprocess

TOP_FEATURES = [
    'Contract_Month-to-month', 'InternetService_Fiber optic',
    'Contract_Two year', 'InternetService_No',
    'PaymentMethod_Electronic check', 'StreamingMovies',
    'OnlineSecurity', 'Contract_One year', 'tenure'
]

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def predict(input_df: pd.DataFrame) -> dict:
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    processed = preprocess(input_df)
    processed[NUM_COLS] = scaler.transform(processed[NUM_COLS])

    for col in TOP_FEATURES:
        if col not in processed.columns:
            processed[col] = 0

    prob = model.predict_proba(processed[TOP_FEATURES])[0][1]
    prediction = model.predict(processed[TOP_FEATURES])[0]

    return {
        'churn': bool(prediction),
        'probability': round(float(prob), 4)
    }