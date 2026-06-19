# Telco Customer Churn Classification

Predikcija odlaska korisnika (churn) telekomunikacione usluge na osnovu demografskih, uslužnih i fakturačkih podataka. SAUSAU predmetni projekat.

## Struktura projekta

```
projekat/
├── data/
│   ├── raw/                      # originalni telco_data.csv
│   └── processed/                # preprocessirani dataset
├── models/                       # exportovani modeli (.pkl) i scaler
├── notebooks/
│   └── 01_eda.ipynb              # EDA, preprocessing, modelovanje
├── reports/
│   └── figures/                  # grafici iz EDA i evaluacije
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── preprocessing.py      # priprema i enkodiranje podataka
│       ├── model.py              # treniranje XGBoost modela
│       ├── evaluate.py           # metrike i konfuziona matrica
│       └── predict.py            # predikcija na novim podacima
├── app.py                        # Streamlit aplikacija
├── pyproject.toml
└── README.md
```

## Zahtevi

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — upravljanje zavisnostima i virtuelnim okruženjem

## Instalacija

Kloniraj repozitorijum i instaliraj zavisnosti:

```bash
git clone <url-repozitorijuma>
cd projekat
uv sync
```

`uv sync` instalira sve zavisnosti definisane u `pyproject.toml` u izolovano virtuelno okruženje.

### Glavne zavisnosti

| Paket | Namena |
|---|---|
| `pandas`, `numpy` | Manipulacija podacima |
| `matplotlib`, `seaborn` | Vizualizacija |
| `scikit-learn` | Preprocessing, modeli, metrike, GridSearchCV |
| `xgboost` | XGBoost klasifikator |
| `streamlit` | Web aplikacija za predikciju |
| `joblib` | Serijalizacija modela |
| `jupyter`, `ipykernel` | Notebook okruženje |

## Pokretanje

### Jupyter notebook (EDA, preprocessing, treniranje modela)

```bash
uv run jupyter notebook
```

Otvori `notebooks/01_eda.ipynb`.

### Streamlit aplikacija

```bash
uv run streamlit run app.py
```

Aplikacija se pokreće na `http://localhost:8501`. Omogućava unos podataka o korisniku i predikciju verovatnoće churna u realnom vremenu.

## Korišćenje modula iz `src/churn`

Svi moduli su pisani kao funkcije koje se mogu uvesti i koristiti nezavisno od notebooka:

```python
from src.churn.preprocessing import preprocess
from src.churn.predict import predict
import pandas as pd

df_raw = pd.read_csv('data/raw/telco_data.csv')
df = preprocess(df_raw)
```

Za predikciju nad jednim novim korisnikom, pogledaj `src/churn/predict.py`.

## Model u produkciji

Aplikacija i `predict.py` koriste **XGBoost** model sa podešenim hiperparametrima (GridSearchCV), treniran na 9 najvažnijih atributa (odabranih putem feature importance analize). Model i `StandardScaler` su sačuvani u `models/` folderu kao `.pkl` fajlovi pomoću `joblib`.

| Fajl | Sadržaj |
|---|---|
| `models/xgb_model.pkl` | Finalni XGBoost model (tuned, top 9 atributa) |
| `models/scaler.pkl` | StandardScaler fitovan na train setu |
| `models/lr_model.pkl` | Logistička regresija (baseline) |
| `models/rf_model.pkl`, `rf_model_tuned.pkl` | Random Forest (default i tuned) |
| `models/xgb_model_default.pkl` | XGBoost sa default hiperparametrima |
| `models/svm_model_tuned.pkl` | SVM (tuned) |

## Napomene

- Notebook koristi `%load_ext autoreload` / `%autoreload 2` kako bi se izmene u `src/churn/*.py` fajlovima automatski učitavale bez restarta kernela.
- Dataset sadrži 11 redova sa nedostajućom vrednošću u `TotalCharges` (korisnici sa `tenure=0`), to je tretirano u `preprocessing.py`.
- Ciljna promenljiva (`Churn`) je neuravnotežena (~73.5% No / ~26.5% Yes), to je kompenzovano korišćenjem `class_weight='balanced'` ili `scale_pos_weight`.
