import pandas as pd

df = pd.read_csv('../data/household_energy.csv')

# Cek rata-rata energi berdasarkan AC
hasil = df.groupby('Has_AC')['Energy_Consumption_kWh'].agg(['mean', 'std', 'count'])
print(hasil)