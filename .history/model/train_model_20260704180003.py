import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

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
# ✅ FITUR DAN TARGET (TANPA PEAK_HOUR)
# =========================
X = df[['Household_Size',
        'Avg_Temperature_C',
        'Has_AC']]  # ← HAPUS Peak_Hours_Usage_kWh

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

print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# ===== TAMPILKAN FEATURE IMPORTANCE =====
print("\n=========================")
print("FEATURE IMPORTANCE")
print("=========================")
features = X.columns
for feat, imp in zip(features, model.feature_importances_):
    print(f"{feat}: {imp:.4f} ({imp*100:.2f}%)")

# =========================
# SIMPAN MODEL
# =========================
joblib.dump(model, 'energy_model.pkl')

print("\nModel berhasil disimpan!")

# 1. Cek training vs testing score
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print(f"Training R²: {train_r2}")
print(f"Testing R²: {test_r2}")

# Jika selisih > 0.05 → Ada overfitting
if abs(train_r2 - test_r2) > 0.05:
    print("⚠️ Kemungkinan Overfitting!")
else:
    print("✅ Model Balanced & Good!")

# 2. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")