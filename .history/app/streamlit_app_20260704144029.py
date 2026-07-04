import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os
import plotly.express as px

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Energi Rumah Tangga",
    layout="centered"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('../data/household_energy.csv')

    # Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # Konversi kategori
    df['Has_AC'] = df['Has_AC'].map({
        'Yes': 1,
        'No': 0
    })

    return df


df = load_data()

# Sampling aman
sample_df = df.sample(
    min(90000, len(df)),
    random_state=42
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load('../model/energy_model.pkl')


if not os.path.exists('../model/energy_model.pkl'):
    st.error(
        "File energy_model.pkl tidak ditemukan. "
        "Silakan training ulang model."
    )
    st.stop()

model = load_model()

# =========================
# HEADER
# =========================
st.title("Dashboard Konsumsi Energi Rumah Tangga")

st.markdown("""
Dashboard ini menampilkan:

- Analisis konsumsi energi rumah tangga
- Visualisasi data
- Machine Learning Random Forest
- Prediksi konsumsi energi
""")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset",
    "📊 Statistik",
    "📈 Visualisasi",
    "🤖 Machine Learning",
    "⚡ Prediksi"
])

# ==================================================
# TAB DATASET
# ==================================================
with tab1:

    st.subheader("Dataset")
    st.dataframe(sample_df.head(50))

    st.write("Jumlah data :", len(sample_df))

    st.markdown("""
        ### Penjelasan Dataset

        Dataset berasal dari Kaggle dengan sekitar 90.000 data.

        Dataset berisi:
        - konsumsi energi rumah tangga,
        - suhu lingkungan,
        - jumlah penghuni,
        - penggunaan AC,
        - penggunaan energi pada jam sibuk.
        """)

# ==================================================
# TAB STATISTIK
# ==================================================
with tab2:

    st.subheader("Statistik")
    st.dataframe(sample_df.describe())
    
    st.markdown("""
    ### Analisa Statistik Data

    Statistik deskriptif digunakan untuk memahami pola dan karakteristik data konsumsi energi rumah tangga.

    Beberapa informasi penting yang diperoleh dari analisis ini antara lain:

    - **Count** → jumlah data pada setiap variabel.
    - **Mean** → rata-rata nilai dari setiap fitur.
    - **Std (Standar Deviasi)** → menunjukkan seberapa besar penyebaran data.
    - **Min dan Max** → nilai terkecil dan terbesar pada dataset.
    - **Quartile (25%, 50%, 75%)** → distribusi dan persebaran data.

    Berdasarkan hasil statistik:
    - Konsumsi energi memiliki variasi yang cukup besar antar rumah tangga.
    - Suhu lingkungan memengaruhi penggunaan energi terutama pada rumah yang menggunakan AC.
    - Jumlah penghuni rumah juga berpengaruh terhadap peningkatan konsumsi listrik.
    - Peak hour usage menunjukkan penggunaan energi tertinggi terjadi pada jam-jam tertentu.
    """)  
    


# ==================================================
# TAB VISUALISASI
# ==================================================
with tab3:

    st.subheader("Heatmap Korelasi")

    corr = sample_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
        Heatmap menunjukkan hubungan antar variabel.

        Nilai:
        - mendekati 1 = hubungan kuat
        - mendekati 0 = hubungan lemah
        """)

    st.subheader("Distribusi Energi")

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    sns.histplot(
        sample_df['Energy_Consumption_kWh'],
        bins=20,
        kde=True,
        ax=ax2
    )

    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("""
        Histogram menunjukkan distribusi konsumsi energi rumah tangga.
        """)
    
    st.subheader("Suhu vs Energi")
    fig3, ax3 = plt.subplots(figsize=(4,3))
    
    sns.scatterplot(
            data=sample_df,
            x='Avg_Temperature_C',
            y='Energy_Consumption_kWh',
            hue='Has_AC',
            alpha=0.5,
            ax=ax3
        )

    ax3.set_xlabel("Suhu (°C)")
    ax3.set_ylabel("Energi")
    
    plt.tight_layout()
    
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown(""" Scatter plot menunjukkan hubungan suhu dengan konsumsi energi rumah tangga. Rumah dengan AC cenderung memiliki konsumsi energi lebih tinggi pada suhu yang sama dibandingkan rumah tanpa AC.""")
    
    st.subheader("Energi Berdasarkan AC")
    
    avg_energy = sample_df.groupby(
            'Has_AC'
        )['Energy_Consumption_kWh'].mean()

    fig4, ax4 = plt.subplots(figsize=(4,3))

    avg_energy.plot(
            kind='bar',
            ax=ax4
        )

    ax4.set_xticklabels([
            'Tanpa AC',
            'Dengan AC'
        ])

    ax4.set_ylabel("Rata-rata Energi")
    plt.tight_layout()

    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("""
       Rumah dengan AC memiliki konsumsi energi lebih tinggi.
        """)
# ==================================================
# TAB MACHINE LEARNING
# ==================================================
with tab4:

    st.subheader("Feature Importance")

    features = [
        'Household_Size',
        'Avg_Temperature_C',
        'Has_AC',
        'Peak_Hours_Usage_kWh'
    ]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })

    fig3, ax3 = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        ax=ax3
    )

    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("""
    ### Penjelasan Feature Importance

    Visualisasi ini menunjukkan tingkat pengaruh setiap variabel terhadap prediksi konsumsi energi.

    Semakin besar nilai importance maka semakin berpengaruh terhadap hasil prediksi.
    """)

# ==================================================
# TAB PREDIKSI
# ==================================================
with tab5:

    st.subheader("Prediksi Konsumsi Energi")

    col1, col2 = st.columns(2)

    with col1:
        household = st.slider(
            "Jumlah Penghuni",
            1,
            10,
            4
        )

        temp = st.slider(
            "Suhu (°C)",
            10,
            40,
            25
        )

    with col2:
        # ✅ UBAH INI
        ac_option = st.selectbox(
            "Menggunakan AC?",
            ["Tidak", "Ya"]
        )
        ac = 1 if ac_option == "Ya" else 0

        peak = st.slider(
            "Peak Hour Usage (kWh)",
            0.0,
            20.0,
            5.0
        )

    input_data = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [ac],
        'Peak_Hours_Usage_kWh': [peak]
    })

    # DEBUG
    st.write(f"**Debug:** AC={ac} ({ac_option}), Household={household}, Temp={temp}, Peak={peak}")

    # Prediksi
    prediction = model.predict(input_data)

    st.success(
        f"Prediksi konsumsi energi: "
        f"{prediction[0]:.4f} kWh"
    )

    st.subheader("Perbandingan: Dengan AC vs Tanpa AC")

    # PERBANDINGAN 1: Dengan Peak Hour Standard
    st.write("**Skenario 1: Dengan Peak Hour Usage Standard (3.0 kWh)**")
    col_compare1, col_compare2 = st.columns(2)

    peak_standard = 3.0

    input_dengan_ac_std = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [1],
        'Peak_Hours_Usage_kWh': [peak_standard]
    })
    pred_dengan_ac_std = model.predict(input_dengan_ac_std)[0]

    input_tanpa_ac_std = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [0],
        'Peak_Hours_Usage_kWh': [peak_standard]
    })
    pred_tanpa_ac_std = model.predict(input_tanpa_ac_std)[0]

    selisih_std = pred_dengan_ac_std - pred_tanpa_ac_std

    with col_compare1:
        st.metric("🔥 Dengan AC", f"{pred_dengan_ac_std:.2f} kWh")

    with col_compare2:
        st.metric("❄️ Tanpa AC", f"{pred_tanpa_ac_std:.2f} kWh")

    st.success(f"✅ Perbedaan: {selisih_std:.2f} kWh ({(selisih_std/pred_tanpa_ac_std)*100:.1f}%)")

    # PERBANDINGAN 2: Dengan Peak Hour Usage Dari Input User
    st.write("\n**Skenario 2: Dengan Peak Hour Usage Input Anda ({peak} kWh)**")
    col_compare3, col_compare4 = st.columns(2)

    input_dengan_ac_user = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [1],
        'Peak_Hours_Usage_kWh': [peak]
    })
    pred_dengan_ac_user = model.predict(input_dengan_ac_user)[0]

    input_tanpa_ac_user = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [0],
        'Peak_Hours_Usage_kWh': [peak]
    })
    pred_tanpa_ac_user = model.predict(input_tanpa_ac_user)[0]

    selisih_user = pred_dengan_ac_user - pred_tanpa_ac_user

    with col_compare3:
        st.metric("🔥 Dengan AC", f"{pred_dengan_ac_user:.2f} kWh")

    with col_compare4:
        st.metric("❄️ Tanpa AC", f"{pred_tanpa_ac_user:.2f} kWh")

    if selisih_user > 0:
        st.success(f"✅ Perbedaan: {selisih_user:.2f} kWh ({(selisih_user/pred_tanpa_ac_user)*100:.1f}%)")
    else:
        st.warning(f"⚠️ Perbedaan: {selisih_user:.2f} kWh - Pengaruh AC minimal saat Peak Hour tinggi")
    
    # Visualisasi perbandingan
    fig_compare, ax_compare = plt.subplots(figsize=(6, 4))
    
    categories = ['Tanpa AC', 'Dengan AC']
    values = [pred_tanpa_ac, pred_dengan_ac]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax_compare.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Tambahkan value di atas bar
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax_compare.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                       f'{value:.2f} kWh', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_compare.set_ylabel("Konsumsi Energi (kWh)", fontsize=12)
    ax_compare.set_title("Perbandingan Konsumsi Energi: Dengan AC vs Tanpa AC", fontsize=13, fontweight='bold')
    ax_compare.set_ylim(0, max(values) * 1.15)
    ax_compare.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_compare)
    plt.close(fig_compare)

    # ===== EVALUASI MODEL =====
    st.subheader("Evaluasi Model")
    
    from sklearn.model_selection import train_test_split
    
    df_full = load_data()
    
    X_full = df_full[
        [
            'Household_Size',
            'Avg_Temperature_C',
            'Has_AC',
            'Peak_Hours_Usage_kWh'
        ]
    ]
    
    y_full = df_full['Energy_Consumption_kWh']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=42
    )
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.3f}")

    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    st.dataframe(pred_df.head(20))

    fig4, ax4 = plt.subplots(figsize=(8, 4))

    ax4.plot(
        pred_df['Actual'].values[:100],
        label='Actual',
        linewidth=2
    )

    ax4.plot(
        pred_df['Predicted'].values[:100],
        label='Predicted',
        linewidth=2
    )

    ax4.set_xlabel("Data Index")
    ax4.set_ylabel("Energi (kWh)")
    ax4.set_title("Actual vs Predicted (Test Set)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    st.pyplot(fig4)
    plt.close(fig4)
    
    fig = px.histogram(df_full, x="Energy_Consumption_kWh")
    st.plotly_chart(fig)