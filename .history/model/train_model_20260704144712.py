import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # ✅ TAMBAH INI
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('../data/household_energy.csv')

# =========================
# DATA CLEANING
# =========================
df = df.dropna()
df = df.drop_duplicates()

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
# ✅ NORMALISASI FITUR
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,  # ✅ GUNAKAN X_SCALED BUKAN X
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

print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# ===== ✅ TAMPILKAN FEATURE IMPORTANCE =====
print("\n=========================")
print("FEATURE IMPORTANCE (AFTER SCALING)")
print("=========================")
features = X.columns
for feat, imp in zip(features, model.feature_importances_):
    print(f"{feat}: {imp:.4f} ({imp*100:.2f}%)")

# =========================
# SIMPAN MODEL & SCALER
# =========================
joblib.dump(model, 'energy_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # ✅ SIMPAN SCALER JUGA!

print("\nModel dan Scaler berhasil disimpan!")