import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prima sirovi DataFrame, vraca ociscen i enkodiran DataFrame
    spreman za modelovanje.
    """
    df = df.copy()  # ne menjamo originalne podatke

    # Uklanjanje irelevantnih kolona
    df = df.drop(columns=['customerID'])

    # Popravka TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Enkodiranje ciljne promenljive
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Binarne kolone
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'gender']
    binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    df[binary_cols] = df[binary_cols].replace(binary_map)

    # Kolone sa 3 vrednosti
    three_val_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    replace_map = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
    df[three_val_cols] = df[three_val_cols].replace(replace_map)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'],
                    drop_first=False, dtype=int)
    
    # Osiguraj da su sve kolone numeričke
    df[binary_cols] = df[binary_cols].astype(int)
    df[three_val_cols] = df[three_val_cols].astype(int)

    return df