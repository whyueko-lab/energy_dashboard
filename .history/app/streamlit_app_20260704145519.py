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

    # ✅ TAMBAHAN: INFO STATS
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Total Data", f"{len(sample_df):,}")
    
    with col_info2:
        st.metric("Jumlah Fitur", len(sample_df.columns))
    
    with col_info3:
        ac_count = (sample_df['Has_AC'] == 1).sum()
        st.metric("Rumah dengan AC", f"{ac_count:,}")

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
    }).sort_values('Importance', ascending=True)  # ✅ Sort untuk lebih jelas

    fig3, ax3 = plt.subplots(figsize=(6, 4))

    bars = sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        ax=ax3
    )
    
    # ✅ Tambahkan angka di bar
    for i, v in enumerate(importance_df['Importance']):
        ax3.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)

    ax3.set_xlabel("Importance Score")
    st.pyplot(fig3)
    plt.close(fig3)

    # ✅ Tampilkan tabel importance
    st.dataframe(importance_df.sort_values('Importance', ascending=False), use_container_width=True)

    st.markdown("""
    ### Penjelasan Feature Importance

    Visualisasi ini menunjukkan tingkat pengaruh setiap variabel terhadap prediksi konsumsi energi.

    Semakin besar nilai importance maka semakin berpengaruh terhadap hasil prediksi:
    
    - **Peak_Hours_Usage_kWh** → Penggunaan pada jam sibuk (dominan)
    - **Avg_Temperature_C** → Suhu lingkungan
    - **Household_Size** → Jumlah penghuni
    - **Has_AC** → Ketersediaan AC
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
        ac_option = st.selectbox(
            "Menggunakan AC?",
            ["Tidak", "Ya"],
            help="Pilih 'Ya' jika rumah menggunakan AC"
        )
        ac = 1 if ac_option == "Ya" else 0

        # ❌ HAPUS PEAK HOUR SLIDER
        # peak = st.slider(...)

    input_data = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [ac]
        # ❌ HAPUS 'Peak_Hours_Usage_kWh'
    })

    # ... sisa code ...

    # ===== EVALUASI MODEL =====
    st.subheader("Evaluasi Model")
    
    from sklearn.model_selection import train_test_split
    
    X_full = df[
        [
            'Household_Size',
            'Avg_Temperature_C',
            'Has_AC',
            'Peak_Hours_Usage_kWh'
        ]
    ]
    
    y_full = df['Energy_Consumption_kWh']
    
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
    
    fig = px.histogram(df, x="Energy_Consumption_kWh")
    st.plotly_chart(fig)