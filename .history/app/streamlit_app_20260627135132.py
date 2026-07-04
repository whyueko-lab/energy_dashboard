import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

    # KONVERSI DATA
    df['Has_AC'] = df['Has_AC'].map({
        'Yes': 1,
        'No': 0
    })

    return df

df = load_data()

# =========================
# SAMPLING DATA
# =========================
sample_df = df.sample(90000, random_state=42)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load('../model/energy_model.pkl')

model = load_model()

# =========================
# JUDUL
# =========================
st.title("Dashboard Konsumsi Energi Rumah Tangga")

st.markdown("""
Dashboard ini menampilkan:
- Analisis konsumsi energi rumah tangga
- Visualisasi data interaktif
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
# TAB 1 - DATASET
# ==================================================
with tab1:

    st.subheader("Dataset")

    st.dataframe(sample_df.head(50))

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
# TAB 2 - STATISTIK
# ==================================================
with tab2:

    st.subheader("Statistik Dataset")

    st.write(sample_df.describe())

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

    Analisis statistik membantu dalam:
    - memahami pola data,
    - mendeteksi anomali,
    - menentukan fitur penting,
    - serta mempersiapkan data sebelum proses Machine Learning dilakukan.
    """)

# ==================================================
# TAB 3 - VISUALISASI
# ==================================================
with tab3:

    # =========================
    # BARIS 1
    # =========================
    col1, col2 = st.columns(2)

    # -------------------------
    # HEATMAP
    # -------------------------
    with col1:

        st.subheader("Heatmap Korelasi")

        corr = sample_df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(6,5))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            ax=ax
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        Heatmap menunjukkan hubungan antar variabel.

        Nilai:
        - mendekati 1 = hubungan kuat
        - mendekati 0 = hubungan lemah
        """)

    # -------------------------
    # HISTOGRAM
    # -------------------------
    with col2:

        st.subheader("Distribusi Energi")

        fig2, ax2 = plt.subplots(figsize=(4,3))

        sns.histplot(
            sample_df['Energy_Consumption_kWh'],
            bins=20,
            kde=True,
            ax=ax2
        )

        ax2.set_xlabel("Energi (kWh)")
        ax2.set_ylabel("Jumlah")

        plt.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("""
        Histogram menunjukkan distribusi konsumsi energi rumah tangga.
        """)

    # =========================
    # BARIS 2
    # =========================
    col3, col4 = st.columns(2)

    # -------------------------
    # SCATTER PLOT
    # -------------------------
    with col3:

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

        st.markdown("""
        Scatter plot menunjukkan hubungan suhu dengan konsumsi energi.
        """)

    # -------------------------
    # BAR CHART
    # -------------------------
    with col4:

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
# TAB 4 - MACHINE LEARNING
# ==================================================
with tab4:

    st.subheader("Feature Importance Random Forest")

    features = [
        'Household_Size',
        'Avg_Temperature_C',
        'Has_AC',
        'Peak_Hours_Usage_kWh'
    ]

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })

    fig5, ax5 = plt.subplots(figsize=(5,3))

    sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        ax=ax5
    )

    plt.tight_layout()

    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown("""
    ### Penjelasan Feature Importance

    Visualisasi ini menunjukkan tingkat pengaruh setiap variabel terhadap prediksi konsumsi energi.

    Semakin besar nilai importance maka semakin berpengaruh terhadap hasil prediksi.
    """)

# ==================================================
# TAB 5 - PREDIKSI
# ==================================================
with tab5:

    st.subheader("Prediksi Konsumsi Energi")

    col5, col6 = st.columns(2)

    with col5:

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

    with col6:

        ac = st.selectbox(
            "Menggunakan AC?",
            [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak"
        )

        peak = st.slider(
            "Peak Hour Usage",
            0.0,
            10.0,
            3.0
        )

    # =========================
    # INPUT DATA
    # =========================
    input_data = pd.DataFrame({
        'Household_Size': [household],
        'Avg_Temperature_C': [temp],
        'Has_AC': [ac],
        'Peak_Hours_Usage_kWh': [peak]
    })

    # ==================================================
    # TAB 5 - PREDIKSI VISUAL
    # ==================================================
with tab5:

    st.subheader("Visualisasi Prediksi Konsumsi Energi")

    # =========================
    # FITUR
    # =========================
    X = sample_df[[
        'Household_Size',
        'Avg_Temperature_C',
        'Has_AC',
        'Peak_Hours_Usage_kWh'
    ]]

    # =========================
    # TARGET ASLI
    # =========================
    y_actual = sample_df['Energy_Consumption_kWh']

    # =========================
    # PREDIKSI MODEL
    # =========================
    y_pred = model.predict(X)

    # =========================
    # EVALUASI MODEL
    # =========================
    mae = mean_absolute_error(y_actual, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_actual, y_pred)
    )

    # =========================
    # METRICS
    # =========================
    col7, col8 = st.columns(2)

    with col7:
        st.metric(
            label="MAE",
            value=f"{mae:.2f}"
        )

    with col8:
        st.metric(
            label="RMSE",
            value=f"{rmse:.2f}"
        )

    st.markdown("""
    ### Penjelasan Evaluasi

    - MAE (Mean Absolute Error)
      menunjukkan rata-rata selisih prediksi dengan data asli.

    - RMSE (Root Mean Squared Error)
      menunjukkan tingkat error model dengan penalti lebih besar untuk kesalahan tinggi.

    Semakin kecil nilai MAE dan RMSE,
    maka model semakin baik.
    """)

    # =========================
    # DATAFRAME
    # =========================
    pred_df = pd.DataFrame({
        'Actual': y_actual,
        'Predicted': y_pred
    })

    st.dataframe(pred_df.head(20))

    # =========================
    # VISUALISASI PREDIKSI
    # =========================
    fig6, ax6 = plt.subplots(figsize=(7,4))

    ax6.plot(
        pred_df['Actual'].values[:100],
        label='Actual'
    )

    ax6.plot(
        pred_df['Predicted'].values[:100],
        label='Predicted'
    )

    ax6.set_xlabel("Data")
    ax6.set_ylabel("Energi (kWh)")
    ax6.set_title("Actual vs Predicted")

    ax6.legend()

    plt.tight_layout()

    st.pyplot(fig6)
    plt.close(fig6)