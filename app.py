# -*- coding: utf-8 -*-
"""
Dashboard Interaktif Harga Emas Batangan Antam (2010-2024)
Dibangun dengan Streamlit, membaca dataset yang sudah bersih
(antam_price_clean.csv) hasil preprocessing di Google Colab.
"""

# ============================================================
# BAGIAN 1: IMPORT LIBRARY DAN PEMUATAN DATA
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="crest")
plt.rcParams["figure.figsize"] = (10, 6)

colors = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd"
}

st.set_page_config(page_title="Dashboard Harga Emas Antam", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("antam_price_clean.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
    df = df.sort_values('Tanggal').reset_index(drop=True)
    return df


df = load_data()

# ============================================================
# BAGIAN 2: REKAYASA FITUR DAN PERHITUNGAN STATISTIK
# ============================================================
df['Tahun'] = df['Tanggal'].dt.year
df['Bulan'] = df['Tanggal'].dt.month
df['Return'] = df['Harga'].pct_change()
df['MA_100'] = df['Harga'].rolling(window=100).mean()
df['Cumulative_Return'] = (1 + df['Return'].fillna(0)).cumprod()
df['zscore'] = (df['Harga'] - df['Harga'].mean()) / df['Harga'].std()

st.title("📊 Dashboard Interaktif Harga Emas Batangan Antam (2010–2024)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Rata-rata", f"Rp{df['Harga'].mean():,.0f}")
col2.metric("Harga Minimum", f"Rp{df['Harga'].min():,.0f}")
col3.metric("Harga Maksimum", f"Rp{df['Harga'].max():,.0f}")
col4.metric("Std Deviasi", f"Rp{df['Harga'].std():,.0f}")

# ============================================================
# BAGIAN 3: ELEMEN INTERAKTIF DAN VISUALISASI
# ============================================================
st.sidebar.header("Filter Dashboard")

tahun_min, tahun_max = int(df['Tahun'].min()), int(df['Tahun'].max())
rentang_tahun = st.sidebar.slider(
    "Pilih Rentang Tahun",
    min_value=tahun_min,
    max_value=tahun_max,
    value=(tahun_min, tahun_max)
)

opsi_visual = [
    "Tren Harga & MA-100",
    "Distribusi Harga (Histogram)",
    "Distribusi Harga (Boxplot)",
    "Rata-rata Tahunan & YoY",
    "Heatmap Tahun vs Bulan",
    "Volatilitas Return Harian",
    "Deteksi Outlier (Z-Score)",
    "Cumulative Return"
]
visual_terpilih = st.sidebar.multiselect(
    "Pilih Visualisasi yang Ditampilkan",
    options=opsi_visual,
    default=opsi_visual
)

df_filtered = df[(df['Tahun'] >= rentang_tahun[0]) & (df['Tahun'] <= rentang_tahun[1])]

# --- Visualisasi 1: Tren Harga dan MA-100 ---
if "Tren Harga & MA-100" in visual_terpilih:
    st.subheader("Tren Harga Emas Antam dan Moving Average 100 Hari")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['Tanggal'], df_filtered['Harga'], color=colors['primary'],
            label='Harga Harian', linewidth=1)
    ax.plot(df_filtered['Tanggal'], df_filtered['MA_100'], color=colors['secondary'],
            label='MA 100 Hari', linewidth=2)
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Harga (Rp/gram)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualisasi 2: Distribusi Harga (Histogram + KDE) ---
if "Distribusi Harga (Histogram)" in visual_terpilih:
    st.subheader("Histogram Distribusi Harga Emas Antam dengan Kurva KDE")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtered['Harga'], kde=True, bins=40, ax=ax)
    ax.set_xlabel('Harga (Rp/gram)')
    ax.set_ylabel('Frekuensi')
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualisasi 3: Boxplot ---
if "Distribusi Harga (Boxplot)" in visual_terpilih:
    st.subheader("Boxplot Distribusi Harga Emas Batangan Antam")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df_filtered["Harga"], ax=ax)
    ax.set_xlabel("Harga (Rp/gram)")
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualisasi 4: Rata-rata Tahunan dan YoY ---
if "Rata-rata Tahunan & YoY" in visual_terpilih:
    st.subheader("Rata-rata Harga Tahunan dan Pertumbuhan YoY")
    rata_tahunan = df_filtered.groupby('Tahun')['Harga'].mean()
    yoy = rata_tahunan.pct_change() * 100
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(rata_tahunan.index, rata_tahunan.values, color=colors['primary'], label='Rata-rata Harga')
    ax1.set_ylabel('Rata-rata Harga (Rp/gram)')
    ax2 = ax1.twinx()
    ax2.plot(yoy.index, yoy.values, color=colors['secondary'], marker='o', label='Pertumbuhan YoY (%)')
    ax2.set_ylabel('Pertumbuhan YoY (%)')
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualisasi 5: Heatmap Tahun vs Bulan ---
if "Heatmap Tahun vs Bulan" in visual_terpilih:
    st.subheader("Heatmap Rata-rata Harga Emas Antam per Tahun dan Bulan")
    pivot = df_filtered.pivot_table(index='Tahun', columns='Bulan', values='Harga', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, ax=ax)
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Tahun')
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualisasi 6: Volatilitas Return Harian ---
if "Volatilitas Return Harian" in visual_terpilih:
    st.subheader("Volatilitas Return Harian Harga Emas Antam")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['Tanggal'], df_filtered['Return'], linewidth=0.7, color=colors['purple'])
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Return Harian')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"Standar deviasi return harian: {df_filtered['Return'].std() * 100:.2f}% per hari")

# --- Visualisasi 7: Deteksi Outlier dengan Z-Score ---
if "Deteksi Outlier (Z-Score)" in visual_terpilih:
    st.subheader("Deteksi Outlier dengan Metode Z-Score")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['Tanggal'], df_filtered['Harga'], color=colors['primary'], linewidth=1, label='Harga')
    outlier_mask = df_filtered['zscore'].abs() > 3
    ax.scatter(df_filtered.loc[outlier_mask, 'Tanggal'], df_filtered.loc[outlier_mask, 'Harga'],
               color=colors['danger'], s=25, label='Outlier (|z| > 3)', zorder=5)
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Harga (Rp/gram)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Jumlah outlier: {outlier_mask.sum()} dari {len(df_filtered)} data "
        f"({outlier_mask.sum() / len(df_filtered) * 100:.2f}%)"
    )

# --- Visualisasi 8: Cumulative Return ---
if "Cumulative Return" in visual_terpilih:
    st.subheader("Cumulative Return Investasi Emas Antam")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['Tanggal'], df_filtered['Cumulative_Return'], color=colors['accent'],
            linewidth=1.5, label='Cumulative Return')
    ax.axhline(1, color='gray', linewidth=1, linestyle='--', label='Titik Impas (Breakeven)')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Cumulative Return (kali lipat)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"Cumulative return akhir periode: {df_filtered['Cumulative_Return'].iloc[-1]:.2f} kali lipat")

