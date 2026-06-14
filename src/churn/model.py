from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

TOP_FEATURES = [
    'Contract_Month-to-month', 'InternetService_Fiber optic',
    'Contract_Two year', 'InternetService_No',
    'PaymentMethod_Electronic check', 'StreamingMovies',
    'OnlineSecurity', 'Contract_One year', 'tenure'
]

def prepare_data(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    model = XGBClassifier(
        scale_pos_weight=5174/1869,
        random_state=42,
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        subsample=0.8
    )
    model.fit(X_train[TOP_FEATURES], y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test[TOP_FEATURES])
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test[TOP_FEATURES])[:, 1]):.4f}")