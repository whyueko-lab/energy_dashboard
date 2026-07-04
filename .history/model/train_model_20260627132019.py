import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('../data/household_energy.csv')

# =========================
# CEK DATA
# =========================
print(df.head())

# =========================
# DATA CLEANING
# =========================

# Mengecek missing value
print("\nJumlah Missing Value:")
print(df.isnull().sum())

# Menghapus missing value
df = df.dropna()

# Mengecek data duplikat
print("\nJumlah Data Duplikat:")
print(df.duplicated().sum())

# Menghapus data duplikat
df = df.drop_duplicates()

print("\nJumlah data setelah cleaning:", len(df))

# =========================
# KONVERSI YES/NO MENJADI ANGKA
# =========================
df['Has_AC'] = df['Has_AC'].map({
    'Yes': 1,
    'No': 0
})

# =========================
# FITUR DAN TARGET
# =========================
X = df[['Household_Size',
        'Avg_Temperature_C',
        'Has_AC',
        'Peak_Hours_Usage_kWh']]

y = df['Energy_Consumption_kWh']

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL RANDOM FOREST
# =========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# TRAINING
model.fit(X_train, y_train)

# =========================
# PREDIKSI
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUASI MODEL
# =========================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=========================")
print("HASIL EVALUASI MODEL")
print("=========================")

print("RMSE :", rmse)
print("R2 Score :", r2)

# =========================
# SIMPAN MODEL
# =========================
joblib.dump(model, 'energy_model.pkl')

print("\nModel berhasil disimpan!")