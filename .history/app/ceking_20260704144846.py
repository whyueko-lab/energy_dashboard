import pandas as pd
import numpy as np

df = pd.read_csv('../data/household_energy.csv')

# Convert Has_AC
df['Has_AC'] = df['Has_AC'].map({'Yes': 1, 'No': 0})

# HITUNG KORELASI
print("=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

corr_matrix = df[['Household_Size', 'Avg_Temperature_C', 'Has_AC', 
                   'Peak_Hours_Usage_kWh', 'Energy_Consumption_kWh']].corr()

print("\nKorelasi dengan Energy_Consumption_kWh:")
print(corr_matrix['Energy_Consumption_kWh'].sort_values(ascending=False))

# HITUNG R² UNTUK SETIAP FITUR INDIVIDUAL
print("\n" + "=" * 60)
print("R² SCORE JIKA HANYA PAKAI SATU FITUR")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['Household_Size', 'Avg_Temperature_C', 'Has_AC', 'Peak_Hours_Usage_kWh']]
y = df['Energy_Consumption_kWh']

for col in X.columns:
    model = LinearRegression()
    X_single = X[[col]]
    model.fit(X_single, y)
    y_pred = model.predict(X_single)
    r2 = r2_score(y, y_pred)
    
    print(f"{col:30s}: R² = {r2:.4f} ({r2*100:.2f}%)")

# VISUALISASI SCATTER PLOT
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, col in enumerate(['Household_Size', 'Avg_Temperature_C', 'Has_AC', 'Peak_Hours_Usage_kWh']):
    row = idx // 2
    col_idx = idx % 2
    ax = axes[row, col_idx]
    
    ax.scatter(df[col], df['Energy_Consumption_kWh'], alpha=0.3, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel('Energy_Consumption_kWh')
    ax.set_title(f'Correlation: {col}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=100)
plt.show()

print("\nGrafik disimpan sebagai 'correlation_analysis.png'")