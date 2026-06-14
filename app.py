import streamlit as st
import pandas as pd
import joblib
from src.churn.preprocessing import preprocess

# Učitavanje modela i scalera
model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

TOP_FEATURES = [
    'Contract_Month-to-month', 'InternetService_Fiber optic',
    'Contract_Two year', 'InternetService_No',
    'PaymentMethod_Electronic check', 'StreamingMovies',
    'OnlineSecurity', 'Contract_One year', 'tenure'
]

NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

st.title('Telco Customer Churn Predictor')
st.write('Unesite podatke o korisniku da biste predvideli da li će otkazati uslugu.')

# Input forma
st.header('Podaci o korisniku')

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Tenure (meseci)', 0, 72, 12)
    monthly_charges = st.slider('Mesečni trošak ($)', 0, 120, 50)
    contract = st.selectbox('Tip ugovora', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet usluga', ['DSL', 'Fiber optic', 'No'])

with col2:
    payment_method = st.selectbox('Način plaćanja', [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    online_security = st.selectbox('Online zaštita', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming filmovi', ['Yes', 'No', 'No internet service'])

# Predikcija
if st.button('Predvidi'):
    # Kreiraj red podataka kao df_raw
    input_data = pd.DataFrame([{
        'customerID': 'dummy',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': 'No',
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': str(tenure * monthly_charges),
        'Churn': 'No'  # placeholder
    }])

    # Preprocessing
    processed = preprocess(input_data)

    # Skaliranje
    processed[NUM_COLS] = scaler.transform(processed[NUM_COLS])

    # Dodaj kolone koje nedostaju (get_dummies mozda nije kreirao sve)
    for col in TOP_FEATURES:
        if col not in processed.columns:
            processed[col] = 0

    # Predikcija
    prob = model.predict_proba(processed[TOP_FEATURES])[0][1]
    prediction = model.predict(processed[TOP_FEATURES])[0]

    st.header('Rezultat')
    if prediction == 1:
        st.error('⚠️ Korisnik će verovatno otkazati uslugu!')
    else:
        st.success('✅ Korisnik će verovatno ostati.')

    st.metric('Verovatnoća churna', f'{prob:.1%}')