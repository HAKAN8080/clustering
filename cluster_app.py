"""
Cluster Analizi — Mağaza Kapasite + Ürün + Fiyat (3D)
Per-Kategori Gruplama | TOP-1-A Format

Clustering Metodları:
1. K-Means Clustering: Makine öğrenmesi tabanlı kümeleme
2. Experiential Scoring: Her ekseni ayrı 3 kümeye böl, ağırlıklı skor hesapla, 9 kümeye ayır
   Skor = X_küme × w_x + Y_küme × w_y + Z_küme × w_z → TOP1...ALL3 (9 küme)
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime

st.set_page_config(page_title="3D Cluster Analizi", page_icon="📊", layout="wide")

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-cluster { padding: 2.5rem 1.5rem 1rem 1.5rem; }

    /* ── SPLASH OVERLAY ─────────────────────────────────────────────── */
    .splash-overlay {
        position: fixed; inset: 0; z-index: 9999;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        animation: splashFadeOut 0.6s ease 3s forwards;
        pointer-events: none;
    }
    @keyframes splashFadeOut {
        0%   { opacity: 1; transform: scale(1); }
        100% { opacity: 0; transform: scale(1.05); pointer-events: none; }
    }

    /* Splash title */
    .splash-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.6rem; font-weight: 700;
        color: #fff; letter-spacing: 2px;
        text-align: center; z-index: 2;
        animation: titlePop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.3s both;
    }
    .splash-title span { color: #60a5fa; }
    @keyframes titlePop {
        0%   { opacity: 0; transform: translateY(30px) scale(0.85); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    .splash-sub {
        font-size: 0.95rem; color: #64748b;
        margin-top: 8px; z-index: 2; letter-spacing: 3px;
        text-transform: uppercase;
        animation: titlePop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.55s both;
    }

    /* Dot ring spinner */
    .splash-ring {
        width: 140px; height: 140px;
        margin-bottom: 36px; position: relative; z-index: 2;
    }
    .splash-ring .dot {
        width: 12px; height: 12px; border-radius: 50%;
        position: absolute; top: 50%; left: 50%;
        transform-origin: 0 0;
    }
    .splash-ring .dot:nth-child(1)  { background:#60a5fa; transform: rotate(0deg)   translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0s; }
    .splash-ring .dot:nth-child(2)  { background:#818cf8; transform: rotate(36deg)  translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.1s; }
    .splash-ring .dot:nth-child(3)  { background:#a78bfa; transform: rotate(72deg)  translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.2s; }
    .splash-ring .dot:nth-child(4)  { background:#c084fc; transform: rotate(108deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.3s; }
    .splash-ring .dot:nth-child(5)  { background:#e879f9; transform: rotate(144deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.4s; }
    .splash-ring .dot:nth-child(6)  { background:#f472b6; transform: rotate(180deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.5s; }
    .splash-ring .dot:nth-child(7)  { background:#fb923c; transform: rotate(216deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.6s; }
    .splash-ring .dot:nth-child(8)  { background:#facc15; transform: rotate(252deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.7s; }
    .splash-ring .dot:nth-child(9)  { background:#4ade80; transform: rotate(288deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.8s; }
    .splash-ring .dot:nth-child(10) { background:#22d3ee; transform: rotate(324deg) translate(58px,-6px); animation: spinDot 1.8s ease-in-out infinite 0.9s; }
    @keyframes spinDot {
        0%, 100% { opacity: 0.25; transform: rotate(var(--r, 0deg)) translate(58px,-6px) scale(0.7); }
        50%      { opacity: 1;    transform: rotate(var(--r, 0deg)) translate(58px,-6px) scale(1.2); }
    }

    /* Floating background particles */
    .splash-particles { position: absolute; inset: 0; overflow: hidden; }
    .splash-particles .p {
        position: absolute; border-radius: 50%;
        opacity: 0; animation: floatUp linear infinite;
    }
    .splash-particles .p:nth-child(1)  { width:4px;  height:4px;  left:10%; background:#60a5fa55; animation-duration:6s; animation-delay:0s; }
    .splash-particles .p:nth-child(2)  { width:6px;  height:6px;  left:25%; background:#818cf855; animation-duration:8s; animation-delay:1s; }
    .splash-particles .p:nth-child(3)  { width:3px;  height:3px;  left:40%; background:#a78bfa55; animation-duration:5s; animation-delay:0.5s; }
    .splash-particles .p:nth-child(4)  { width:5px;  height:5px;  left:55%; background:#4ade8055; animation-duration:7s; animation-delay:2s; }
    .splash-particles .p:nth-child(5)  { width:4px;  height:4px;  left:70%; background:#f472b655; animation-duration:6.5s; animation-delay:0.8s; }
    .splash-particles .p:nth-child(6)  { width:7px;  height:7px;  left:82%; background:#60a5fa44; animation-duration:9s; animation-delay:1.5s; }
    .splash-particles .p:nth-child(7)  { width:3px;  height:3px;  left:92%; background:#22d3ee55; animation-duration:5.5s; animation-delay:0.3s; }
    .splash-particles .p:nth-child(8)  { width:5px;  height:5px;  left:18%; background:#facc1555; animation-duration:7.5s; animation-delay:1.8s; }
    .splash-particles .p:nth-child(9)  { width:4px;  height:4px;  left:60%; background:#fb923c55; animation-duration:6s; animation-delay:2.5s; }
    .splash-particles .p:nth-child(10) { width:6px;  height:6px;  left:35%; background:#c084fc55; animation-duration:8.5s; animation-delay:0.2s; }
    @keyframes floatUp {
        0%   { opacity: 0; transform: translateY(100vh) rotate(0deg); }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { opacity: 0; transform: translateY(-120px) rotate(360deg); }
    }

    /* ── APP HEADER ─────────────────────────────────────────────────── */
    .app-main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #2F5496 100%);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(15,23,42,0.25);
    }
    .app-main-header::before {
        content: '';
        position: absolute; top: -40%; right: -10%;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(96,165,250,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-main-header::after {
        content: '';
        position: absolute; bottom: -30%; left: 20%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-main-header h1 {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.65rem; font-weight: 700;
        color: #fff; margin: 0; position: relative; z-index: 1;
        letter-spacing: 0.5px;
    }
    .app-main-header h1 span { color: #60a5fa; }
    .app-main-header .header-sub {
        font-size: 0.82rem; color: #64748b;
        margin-top: 4px; position: relative; z-index: 1;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .header-badge-row {
        position: absolute; top: 50%; right: 28px;
        transform: translateY(-50%); z-index: 1;
        display: flex; gap: 8px; align-items: center;
    }
    .hdr-badge {
        font-size: 10px; font-weight: 700; letter-spacing: 1px;
        padding: 4px 10px; border-radius: 20px;
    }
    .hdr-badge.top { background: rgba(74,222,128,0.2); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .hdr-badge.mid { background: rgba(250,204,21,0.2); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
    .hdr-badge.all { background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }

    /* ── EMPTY STATE ────────────────────────────────────────────────── */
    .empty-state-wrap {
        text-align: center; padding: 40px 20px 30px;
    }
    .empty-state-wrap .empty-title {
        font-size: 1.1rem; font-weight: 600; color: #1e293b;
        margin-top: 16px; margin-bottom: 4px;
    }
    .empty-state-wrap .empty-sub {
        font-size: 0.82rem; color: #64748b;
    }

    .section-header {
        background: #C00000;
        color: white;
        padding: 9px 14px;
        font-weight: bold;
        font-size: 13px;
        margin: 0;
        border-radius: 5px;
        border: 1px solid #a00;
    }
    .section-header-blue {
        background: #2F5496;
        color: white;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 14px;
        margin: 0;
        border-radius: 5px;
        border: 1px solid #1e3a6e;
        text-align: center;
    }

    .result-table {
        border-collapse: collapse;
        font-size: 12.5px;
        width: 100%;
        margin-top: 6px;
    }
    .result-table th {
        background: #2F5496;
        color: white;
        padding: 7px 10px;
        border: 1px solid #999;
        text-align: center;
        font-weight: 600;
        font-size: 11.5px;
    }
    .result-table td {
        padding: 5px 10px;
        border: 1px solid #D9D9D9;
        text-align: center;
    }
    .result-table tr:nth-child(even) td { background: #F2F2F2; }
    .result-table tr:hover td { background: #E8F0FE; }

    .badge-top  { background:#C6EFCE; color:#006100; padding:2px 9px; border-radius:11px; font-weight:bold; font-size:11px; }
    .badge-mid  { background:#FFEB9C; color:#9C5700; padding:2px 9px; border-radius:11px; font-weight:bold; font-size:11px; }
    .badge-all  { background:#FFC7CE; color:#9C0006; padding:2px 9px; border-radius:11px; font-weight:bold; font-size:11px; }

    .legend-box {
        background: #f0f4ff;
        border: 1px solid #c5d5f0;
        border-radius: 7px;
        padding: 10px 14px;
        font-size: 12px;
        margin: 6px 0 10px 0;
        line-height: 1.7;
    }
    .legend-box b { color: #2F5496; }

    hr { margin: 8px 0; border-color: #E0E0E0; }

    .stButton > button {
        background: #2F5496 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton > button:hover { background: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)


# ─── CORE: CLUSTERING FUNCTIONS ─────────────────────────────────────────────

def load_data(uploaded_file):
    """Excel / CSV yükle"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def kmeans_global(df, attribute_cols, n_clusters, desc=True):
    """
    Global K-Means — kapasite gruplama (tüm mağazalar üzerinde bir kez).
    desc=True → 1=en büyük (TOP), desc=False → 1=en küçük (ALL)
    """
    X = df[attribute_cols].fillna(df[attribute_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    means = {c: X.iloc[clusters == c].mean().mean() for c in range(n_clusters)}
    sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def kmeans_per_category(df, kategori_col, metric_col, n_clusters,
                        label_type='numeric', desc=True):
    """
    Kategori bazında ayrı K-Means.
    desc=True → 1/A = en büyük, desc=False → 1/A = en küçük
    """
    result = pd.Series(index=df.index, dtype=object)

    for kategori in df[kategori_col].unique():
        mask = df[kategori_col] == kategori
        subset = df.loc[mask, metric_col]

        # Yeterli veri kontrolü
        non_null = subset.dropna()
        if len(non_null) < 2:
            result.loc[mask] = 1 if label_type == 'numeric' else 'A'
            continue

        actual_clusters = min(n_clusters, len(non_null.unique()))
        if actual_clusters < 2:
            result.loc[mask] = 1 if label_type == 'numeric' else 'A'
            continue

        X = subset.fillna(subset.mean()).values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Düşük → yüksek sıralama (clusters zaten subset boyutunda)
        means = {c: subset.values[clusters == c].mean() for c in range(actual_clusters)}
        sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
        mapping = {old: new for new, old in enumerate(sorted_c)}
        sorted_clusters = np.array([mapping[c] for c in clusters])

        if label_type == 'numeric':
            result.loc[mask] = sorted_clusters + 1          # 1, 2, 3…
        else:
            labels = [chr(65 + i) for i in range(actual_clusters)]  # A, B, C…
            result.loc[mask] = [labels[c] for c in sorted_clusters]

    return result


def get_kapasite_label(grup_num, total):
    """Kapasite grup numarası → TOP / MID / ALL (1=en büyük=TOP)"""
    if total == 2:
        return 'TOP' if grup_num == 1 else 'ALL'
    elif total == 3:
        return {1: 'TOP', 2: 'MID', 3: 'ALL'}.get(grup_num, str(grup_num))
    else:
        if grup_num == 1:
            return 'TOP'
        elif grup_num == total:
            return 'ALL'
        else:
            return 'MID'


def compute_unified_groups(df, kap_x_cols, urun_metric_col, urun_fiyat_col,
                           w_kap, w_urun, w_fiyat):
    """
    Ağırlıklı birleşik skor → KMeans ile 9 doğal grup: TOP1…ALL3
    Büyükten küçüğe: TOP1 > TOP2 > TOP3 > MID1 > MID2 > MID3 > ALL1 > ALL2 > ALL3
    """
    from sklearn.preprocessing import MinMaxScaler

    kap_raw = df[kap_x_cols].fillna(0).mean(axis=1).values.reshape(-1, 1)
    urun_raw = df[urun_metric_col].fillna(0).values.reshape(-1, 1)
    fiyat_raw = df[urun_fiyat_col].fillna(0).values.reshape(-1, 1)

    kap_n = MinMaxScaler().fit_transform(kap_raw).flatten()
    urun_n = MinMaxScaler().fit_transform(urun_raw).flatten()
    fiyat_n = MinMaxScaler().fit_transform(fiyat_raw).flatten()

    total_w = w_kap + w_urun + w_fiyat
    if total_w == 0:
        total_w = 1.0
    score = (w_kap / total_w) * kap_n + (w_urun / total_w) * urun_n + (w_fiyat / total_w) * fiyat_n

    # KMeans ile doğal gruplar (eşit dağılım zorlamaz)
    n_clusters = min(9, len(np.unique(score)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(score.reshape(-1, 1))

    # Kümeleri skor ortalamasına göre büyükten küçüğe sırala
    centroids = {c: score[clusters == c].mean() for c in range(n_clusters)}
    sorted_c = sorted(centroids.keys(), key=lambda x: centroids[x], reverse=True)
    mapping = {old: new for new, old in enumerate(sorted_c)}
    sorted_clusters = np.array([mapping[c] for c in clusters])

    group_order = ['TOP1', 'TOP2', 'TOP3', 'MID1', 'MID2', 'MID3', 'ALL1', 'ALL2', 'ALL3']
    unified = np.array([group_order[min(c, 8)] for c in sorted_clusters])

    return score, unified


def assign_experiential_cluster(values, n_clusters=3, desc=True):
    """
    Experiential Scoring için tek bir ekseni n_clusters kümeye böl.
    Quantile tabanlı bölme yapar (eşit sayıda eleman her grupta).
    desc=True → yüksek değer = küme 1 (TOP)
    desc=False → düşük değer = küme 1
    Döndürür: 1, 2, 3, ... şeklinde küme numaraları
    """
    values = pd.Series(values).fillna(pd.Series(values).mean())

    # Quantile tabanlı kümeleme
    try:
        clusters = pd.qcut(values, q=n_clusters, labels=False, duplicates='drop')
        actual_clusters = clusters.max() + 1
    except ValueError:
        # Yeterli unique değer yoksa
        clusters = pd.Series([0] * len(values))
        actual_clusters = 1

    if desc:
        # Yüksek değer = 1 (TOP)
        clusters = actual_clusters - clusters
    else:
        # Düşük değer = 1
        clusters = clusters + 1

    return clusters.values


def experiential_scoring_clustering(df, kap_x_cols, urun_metric_col, urun_fiyat_col,
                                     w_kap, w_urun, w_fiyat, n_clusters=3, desc=True):
    """
    Experiential Scoring Clustering Method:
    1. Her ekseni (X, Y, Z) ayrı ayrı 3 kümeye böl (1, 2, 3)
    2. Ağırlıklı ortalama hesapla: weighted_avg = x_küme * w_x + y_küme * w_y + z_küme * w_z
    3. Bu skoru 9 kümeye böl: TOP1, TOP2, TOP3, MID1, MID2, MID3, ALL1, ALL2, ALL3

    Örnek: x_küme=2, y_küme=1, z_küme=2, ağırlıklar=%30,%40,%30
    weighted_avg = 2*0.3 + 1*0.4 + 2*0.3 = 1.6
    Final küme index = round((1.6 - 1) / 2 * 8) = round(2.4) = 2 → TOP3
    """
    # Kapasite X değeri (birden fazla kolon varsa ortalama)
    kap_values = df[kap_x_cols].fillna(0).mean(axis=1).values
    urun_values = df[urun_metric_col].fillna(0).values
    fiyat_values = df[urun_fiyat_col].fillna(0).values

    # Her ekseni ayrı ayrı 3 kümeye böl (1, 2, 3)
    kap_clusters = assign_experiential_cluster(kap_values, n_clusters, desc)
    urun_clusters = assign_experiential_cluster(urun_values, n_clusters, desc)
    fiyat_clusters = assign_experiential_cluster(fiyat_values, n_clusters, desc)

    # Ağırlıkları normalize et
    total_w = w_kap + w_urun + w_fiyat
    if total_w == 0:
        total_w = 1.0
    w_kap_n = w_kap / total_w
    w_urun_n = w_urun / total_w
    w_fiyat_n = w_fiyat / total_w

    # Ağırlıklı ortalama (1 ile 3 arasında değer)
    weighted_score = (kap_clusters * w_kap_n +
                      urun_clusters * w_urun_n +
                      fiyat_clusters * w_fiyat_n)

    # Ağırlıklı skoru 9 kümeye böl (0-8 arası index)
    # weighted_score: 1.0 (min) - 3.0 (max) aralığında
    # (score - 1) / 2 → 0.0 - 1.0 aralığına normalize
    # * 8 → 0 - 8 aralığına scale
    normalized = (weighted_score - 1.0) / 2.0  # 0-1 arası
    cluster_index = np.round(normalized * 8).astype(int)
    cluster_index = np.clip(cluster_index, 0, 8)

    # 9 grup: TOP1 (en yüksek) → ALL3 (en düşük)
    group_order = ['TOP1', 'TOP2', 'TOP3', 'MID1', 'MID2', 'MID3', 'ALL1', 'ALL2', 'ALL3']
    unified_labels = np.array([group_order[i] for i in cluster_index])

    return (weighted_score, unified_labels,
            kap_clusters, urun_clusters, fiyat_clusters)


def render_splash():
    """Sadece ilk yükleme — session'a flag koy"""
    if 'splash_shown' not in st.session_state:
        st.session_state.splash_shown = True
        st.markdown("""
        <div class="splash-overlay">
            <div class="splash-particles">
                <div class="p"></div><div class="p"></div><div class="p"></div>
                <div class="p"></div><div class="p"></div><div class="p"></div>
                <div class="p"></div><div class="p"></div><div class="p"></div>
                <div class="p"></div>
            </div>
            <div class="splash-ring">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                <div class="dot"></div>
            </div>
            <div class="splash-title">3D <span>Cluster</span> Analizi</div>
            <div class="splash-sub">K-Means & Experiential Scoring</div>
        </div>
        """, unsafe_allow_html=True)


def render_demo_3d():
    """Boş state'te demo animated 3D scatter — gerçek veri olmadan"""
    np.random.seed(42)
    n = 120
    # 3 küme oluştur
    c1 = np.random.randn(n, 3) * 12 + np.array([30, 20, 50])
    c2 = np.random.randn(n, 3) * 10 + np.array([70, 60, 25])
    c3 = np.random.randn(n, 3) * 11 + np.array([50, 80, 70])

    labels = (['TOP-1-A'] * n) + (['MID-2-B'] * n) + (['ALL-3-C'] * n)
    xs = np.concatenate([c1[:, 0], c2[:, 0], c3[:, 0]])
    ys = np.concatenate([c1[:, 1], c2[:, 1], c3[:, 1]])
    zs = np.concatenate([c1[:, 2], c2[:, 2], c3[:, 2]])

    import plotly.graph_objects as go

    colors = {'TOP-1-A': '#60a5fa', 'MID-2-B': '#4ade80', 'ALL-3-C': '#f472b6'}
    fig = go.Figure()
    for label in ['TOP-1-A', 'MID-2-B', 'ALL-3-C']:
        mask = np.array([l == label for l in labels])
        fig.add_trace(go.Scatter3d(
            x=xs[mask], y=ys[mask], z=zs[mask],
            mode='markers', name=label,
            marker=dict(size=5, color=colors[label], opacity=0.75,
                        line=dict(width=0.5, color='white'))
        ))

    fig.update_layout(
        height=380,
        scene=dict(
            xaxis_title='Kapasite', yaxis_title='Ürün Performans', zaxis_title='Fiyat',
            xaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155', zerolinecolor='#334155'),
            yaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155', zerolinecolor='#334155'),
            zaxis=dict(backgroundcolor='#1e293b', gridcolor='#334155', zerolinecolor='#334155'),
            camera=dict(eye=dict(x=1.4, y=1.2, z=0.9))
        ),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(color='#94a3b8', size=11),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.08,
                    font=dict(size=11), bgcolor='rgba(15,23,42,0)'),
        margin=dict(l=0, r=0, t=10, b=40)
    )
    # Auto-rotate JS inject
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <script>
    (function(){
        var plots = document.querySelectorAll('.plotly');
        plots.forEach(function(p){
            var el = p.querySelector('.js-plotly-plot') || p;
            if(window.Plotly && el && el._fullLayout){
                var angle = 0;
                setInterval(function(){
                    angle += 2;
                    var rad = angle * Math.PI / 180;
                    var eye = { x: 1.8*Math.cos(rad), y: 1.8*Math.sin(rad), z: 0.7 };
                    window.Plotly.relayout(el, {scene:{camera:{eye:eye}}});
                }, 80);
            }
        });
    })();
    </script>
    """, unsafe_allow_html=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # Splash — sadece ilk yükleme
    render_splash()

    # Session state init
    for key in ['kapasite_df', 'urun_df', 'kapasite_results', 'final_results', 'config']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Ana Başlık ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-main-header">
        <h1>📊 3D <span>Cluster</span> Analizi</h1>
        <div class="header-sub">Mağaza · Ürün · Fiyat &nbsp;|&nbsp; K-Means & Experiential Scoring</div>
        <div class="header-badge-row">
            <span class="hdr-badge top">TOP</span>
            <span class="hdr-badge mid">MID</span>
            <span class="hdr-badge all">ALL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2.5])

    # ══════════════════════════════════════════════════════════════════════════
    # SOL PANEL — INPUT
    # ══════════════════════════════════════════════════════════════════════════
    with col_left:

        # ─── KAPASITE ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📦 Kapasite — Mağaza Gruplama (Global)</div>',
                    unsafe_allow_html=True)

        uploaded_kap = st.file_uploader(
            "kapasite.xlsx", type=['xlsx', 'xls', 'csv'],
            key='kap_upload', label_visibility="collapsed"
        )
        if uploaded_kap:
            file_id = uploaded_kap.name + str(uploaded_kap.size)
            if st.session_state.get('_kap_file_id') != file_id:
                st.session_state.kapasite_df = load_data(uploaded_kap)
                st.session_state._kap_file_id = file_id

        # Defaults
        kap_label      = None
        kap_attrs      = []
        kap_grup_sayisi = 3

        if st.session_state.kapasite_df is not None:
            df_k           = st.session_state.kapasite_df
            all_cols_k     = df_k.columns.tolist()
            numeric_cols_k = df_k.select_dtypes(include=[np.number]).columns.tolist()

            kap_label = st.selectbox("🏷️ Mağaza Kolonu", options=all_cols_k, key='kap_label')

            available_k = [c for c in numeric_cols_k if c != kap_label]
            kap_attrs   = st.multiselect(
                "📊 Kapasite Attributeları (X-eksen)",
                options=available_k,
                default=available_k[:min(2, len(available_k))],
                key='kap_attrs'
            )
            kap_grup_sayisi = st.number_input(
                "Grup Sayısı", min_value=2, max_value=10, value=3, key='kap_grup'
            )
            st.caption(f"✓ {len(df_k)} mağaza yüklendi")
        else:
            st.caption("📁 kapasite.xlsx yükleyin")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── ÜRÜN ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Ürün — Per-Kategori Gruplama</div>',
                    unsafe_allow_html=True)

        uploaded_urun = st.file_uploader(
            "ürün_data.xlsx", type=['xlsx', 'xls', 'csv'],
            key='urun_upload', label_visibility="collapsed"
        )
        if uploaded_urun:
            file_id = uploaded_urun.name + str(uploaded_urun.size)
            if st.session_state.get('_urun_file_id') != file_id:
                st.session_state.urun_df = load_data(uploaded_urun)
                st.session_state._urun_file_id = file_id

        # Defaults
        urun_magaza_col   = None
        urun_urun_col     = None
        urun_kategori_col = None
        urun_metric_col   = None
        urun_fiyat_col    = None
        urun_grup_sayisi  = 3
        fiyat_grup_sayisi = 3

        if st.session_state.urun_df is not None:
            df_u           = st.session_state.urun_df
            all_cols_u     = df_u.columns.tolist()
            numeric_cols_u = df_u.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_u     = df_u.select_dtypes(include=['object', 'category']).columns.tolist()

            # Mağaza + Ürün kolonu
            urun_magaza_col = st.selectbox("🏪 Mağaza Kolonu", options=all_cols_u, key='u_magaza')
            urun_urun_col   = st.selectbox(
                "📦 Ürün Kolonu",
                options=[c for c in all_cols_u if c != urun_magaza_col],
                key='u_urun'
            )

            # Kategori kolonu (string/categorical) — sadece Mağaza kolonu hariç
            cat_options = [c for c in cat_cols_u if c != urun_magaza_col]
            if not cat_options:
                cat_options = [c for c in all_cols_u
                               if c != urun_magaza_col and c not in numeric_cols_u]

            if cat_options:
                urun_kategori_col = st.selectbox(
                    "🏷️ Kategori Kolonu", options=cat_options, key='u_kategori'
                )
            else:
                st.warning("⚠️ Kategori (string) kolonu bulunamadı — dosyanızı kontrol edin.")

            # Ürün Metrik (Y-eksen) + Fiyat (Z-eksen)
            used = [urun_magaza_col, urun_urun_col]
            if urun_kategori_col:
                used.append(urun_kategori_col)

            metric_options = [c for c in numeric_cols_u if c not in used]

            if len(metric_options) >= 2:
                urun_metric_col = st.selectbox(
                    "📈 Ürün Metrik — Y eksen", options=metric_options, key='u_metric'
                )
                fiyat_options   = [c for c in metric_options if c != urun_metric_col]
                urun_fiyat_col  = st.selectbox(
                    "💰 Fiyat Kolonu — Z eksen", options=fiyat_options, key='u_fiyat'
                )
            elif len(metric_options) == 1:
                urun_metric_col = metric_options[0]
                st.warning("⚠️ Fiyat kolonu için yeterli sayısal kolon yok.")
            else:
                st.warning("⚠️ Sayısal kolon bulunamadı.")

            # Grup sayıları (satır üstünde)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                urun_grup_sayisi = st.number_input(
                    "Ürün Grup (1/2/3)", min_value=2, max_value=10, value=3, key='u_grup'
                )
            with col_g2:
                fiyat_grup_sayisi = st.number_input(
                    "Fiyat Grup (A/B/C)", min_value=2, max_value=10, value=3, key='f_grup'
                )

            n_kat = df_u[urun_kategori_col].nunique() if urun_kategori_col else 0
            st.caption(f"✓ {len(df_u)} satır | {n_kat} kategori")
        else:
            st.caption("📁 ürün_data.xlsx yükleyin")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── CLUSTERING METODU ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🧮 Clustering Metodu</div>',
                    unsafe_allow_html=True)
        clustering_method = st.selectbox(
            "Metod Seçin",
            options=['K-Means Clustering', 'Experiential Scoring'],
            key='clustering_method',
            help="K-Means: Makine öğrenmesi tabanlı kümeleme\nExperiential Scoring: Her ekseni ayrı böl, ağırlıklı ortalamayla final küme hesapla"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── SIRALAMA TERCİHİ ────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔃 Sıralama Yönü</div>',
                    unsafe_allow_html=True)
        siralama_yon = st.radio(
            "Gruplama sıralaması",
            options=['Büyükten Küçüğe (DESC)', 'Küçükten Büyüğe (ASC)'],
            index=0, key='siralama_yon', horizontal=True,
            label_visibility="collapsed"
        )
        desc_order = siralama_yon.startswith('Büyük')

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── AĞIRLIKLAR ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">⚖️ Ağırlıklar — Birleşik Gruplama</div>',
                    unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_kapasite = st.slider("X Ekseni", 0.0, 1.0, 0.4, 0.05, key='w_kap')
        with col_w2:
            w_urun = st.slider("Y Ekseni", 0.0, 1.0, 0.3, 0.05, key='w_urun')
        with col_w3:
            w_fiyat = st.slider("Z Ekseni", 0.0, 1.0, 0.3, 0.05, key='w_fiyat')

        total_w = w_kapasite + w_urun + w_fiyat
        if total_w > 0:
            st.caption(
                f"Normalize: X {w_kapasite/total_w:.0%} · "
                f"Y {w_urun/total_w:.0%} · Z {w_fiyat/total_w:.0%}"
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ─── GRUPLA BUTONU ───────────────────────────────────────────────────
        btn_disabled = (
            st.session_state.kapasite_df is None
            or st.session_state.urun_df is None
            or len(kap_attrs) == 0
            or urun_metric_col is None
            or urun_fiyat_col is None
            or urun_kategori_col is None
        )

        if st.button("🚀 Grupla ve Birleştir",
                      disabled=btn_disabled, use_container_width=True, type="primary"):
            with st.spinner("Gruplama yapılıyor…"):

                # ── STEP 1: Kapasite gruplama (GLOBAL) ──────────────────────
                kap_df = st.session_state.kapasite_df.copy()
                kap_df['_Kap_Grup_Num'] = kmeans_global(
                    kap_df, kap_attrs, kap_grup_sayisi, desc=desc_order
                )
                kap_df['Kapasite_Grubu'] = kap_df['_Kap_Grup_Num'].apply(
                    lambda x: get_kapasite_label(x, kap_grup_sayisi)
                )
                st.session_state.kapasite_results = kap_df

                # ── STEP 2: Ürün — PER KATEGORI gruplama ─────────────────────
                urun_df = st.session_state.urun_df.copy()

                urun_df['Urun_Grubu']  = kmeans_per_category(
                    urun_df, urun_kategori_col, urun_metric_col,
                    urun_grup_sayisi, 'numeric', desc=desc_order
                )
                urun_df['Fiyat_Grubu'] = kmeans_per_category(
                    urun_df, urun_kategori_col, urun_fiyat_col,
                    fiyat_grup_sayisi, 'alpha', desc=desc_order
                )

                # ── STEP 3: Kapasite join (label + X-eksen değerleri) ─────────
                # kap_attrs değerlerini _Kap_X_ prefix ile taşı → kolon çarpışma yok
                join_cols   = [kap_label, 'Kapasite_Grubu'] + kap_attrs
                kap_join    = kap_df[join_cols].copy()
                rename_map  = {col: f'_Kap_X_{col}' for col in kap_attrs}
                kap_join    = kap_join.rename(columns={**rename_map, kap_label: urun_magaza_col})

                urun_df = urun_df.merge(kap_join, on=urun_magaza_col, how='left')
                urun_df['Kapasite_Grubu'] = urun_df['Kapasite_Grubu'].fillna('?')

                unmatched = urun_df['Kapasite_Grubu'].eq('?').sum()

                # ── STEP 4: Kombine Grup → TOP-1-A ──────────────────────────
                urun_df['Kombine_Grup'] = (
                    urun_df['Kapasite_Grubu'].astype(str) + '-' +
                    urun_df['Urun_Grubu'].astype(str)   + '-' +
                    urun_df['Fiyat_Grubu'].astype(str)
                )

                # ── STEP 5: Ağırlıklı Birleşik Gruplama ─────────────────────
                kap_x_cols_list = [f'_Kap_X_{c}' for c in kap_attrs]

                if clustering_method == 'K-Means Clustering':
                    # K-Means tabanlı birleşik gruplama
                    score, unified = compute_unified_groups(
                        urun_df, kap_x_cols_list,
                        urun_metric_col, urun_fiyat_col,
                        w_kapasite, w_urun, w_fiyat
                    )
                    urun_df['Agirlikli_Skor'] = score
                    urun_df['Birlesik_Grup'] = unified
                    urun_df['Exp_Kap_Kume'] = None
                    urun_df['Exp_Urun_Kume'] = None
                    urun_df['Exp_Fiyat_Kume'] = None
                else:
                    # Experiential Scoring tabanlı birleşik gruplama
                    (score, unified,
                     exp_kap, exp_urun, exp_fiyat) = experiential_scoring_clustering(
                        urun_df, kap_x_cols_list,
                        urun_metric_col, urun_fiyat_col,
                        w_kapasite, w_urun, w_fiyat,
                        n_clusters=3, desc=desc_order
                    )
                    urun_df['Agirlikli_Skor'] = score
                    urun_df['Birlesik_Grup'] = unified
                    urun_df['Exp_Kap_Kume'] = exp_kap
                    urun_df['Exp_Urun_Kume'] = exp_urun
                    urun_df['Exp_Fiyat_Kume'] = exp_fiyat

                # ── Session'a kaydet ──────────────────────────────────────────
                st.session_state.final_results = urun_df
                st.session_state.config = {
                    'urun_magaza_col':   urun_magaza_col,
                    'urun_urun_col':     urun_urun_col,
                    'urun_kategori_col': urun_kategori_col,
                    'urun_metric_col':   urun_metric_col,
                    'urun_fiyat_col':    urun_fiyat_col,
                    'kap_label':         kap_label,
                    'kap_attrs':         kap_attrs,
                    'kap_x_cols':        [f'_Kap_X_{c}' for c in kap_attrs],
                    'kap_x_labels':      {f'_Kap_X_{c}': c for c in kap_attrs},
                    'unmatched':         unmatched,
                    'w_kapasite':        w_kapasite,
                    'w_urun':            w_urun,
                    'w_fiyat':           w_fiyat,
                    'clustering_method': clustering_method,
                }

                if unmatched > 0:
                    st.warning(f"⚠️ {unmatched} satırda mağaza eşleştirme yapılamadı — '?' olarak bırakıldı.")
                else:
                    st.success("✅ Gruplama tamamlandı!")

    # ══════════════════════════════════════════════════════════════════════════
    # SAĞ PANEL — SONUÇLAR
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
        st.markdown(
            '<div class="section-header-blue">Sonuç — 3D Cluster Analizi</div>',
            unsafe_allow_html=True
        )

        if st.session_state.final_results is not None and st.session_state.config is not None:
            results = st.session_state.final_results.copy()
            cfg     = st.session_state.config

            urun_magaza_col   = cfg['urun_magaza_col']
            urun_urun_col     = cfg['urun_urun_col']
            urun_kategori_col = cfg['urun_kategori_col']
            urun_metric_col   = cfg['urun_metric_col']
            urun_fiyat_col    = cfg['urun_fiyat_col']
            kap_x_cols        = cfg['kap_x_cols']
            kap_x_labels      = cfg['kap_x_labels']

            # ── Legenda ──────────────────────────────────────────────────────
            method_name = cfg.get('clustering_method', 'K-Means Clustering')
            if method_name == 'K-Means Clustering':
                st.markdown("""
                <div class="legend-box">
                    <b>Metod: K-Means Clustering</b><br>
                    <b>Format → TOP-1-A</b><br>
                    <b>TOP / MID / ALL</b> — Mağaza Kapasite grubu (global, ürün bağımsız)<br>
                    <b>1 / 2 / 3</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Ürün Performans grubu (<i>her kategori içinde ayrı</i>)<br>
                    <b>A / B / C</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Fiyat Seviyesi grubu&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<i>her kategori içinde ayrı</i>)<br>
                    Birleşik grup: Ağırlıklı skor üzerinden K-Means ile 9 grup (TOP1...ALL3)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legend-box">
                    <b>Metod: Experiential Scoring</b><br>
                    <b>Format → TOP-1-A</b><br>
                    <b>TOP / MID / ALL</b> — Mağaza Kapasite grubu (global, ürün bağımsız)<br>
                    <b>1 / 2 / 3</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Ürün Performans grubu (<i>her kategori içinde ayrı</i>)<br>
                    <b>A / B / C</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Fiyat Seviyesi grubu&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<i>her kategori içinde ayrı</i>)<br>
                    <hr style="margin:5px 0; border-color:#ddd;">
                    <b>Birleşik Grup (9 Küme):</b> Her eksen ayrı 3 kümeye bölünür (1,2,3), sonra:<br>
                    Skor = X_küme × {cfg['w_kapasite']:.0%} + Y_küme × {cfg['w_urun']:.0%} + Z_küme × {cfg['w_fiyat']:.0%}<br>
                    → TOP1, TOP2, TOP3, MID1, MID2, MID3, ALL1, ALL2, ALL3
                </div>
                """, unsafe_allow_html=True)

            # ── Kategori filter + X-eksen seçimi ─────────────────────────────
            kategoriler = sorted(results[urun_kategori_col].unique())
            row_top = st.columns([1.8, 1.2])

            with row_top[0]:
                seçilen_kategori = st.selectbox(
                    "🏷️ Kategori Filtre",
                    options=['🔄 Tümü'] + kategoriler,
                    key='kat_filter'
                )
            with row_top[1]:
                if len(kap_x_cols) > 1:
                    kapasite_x_col = st.selectbox(
                        "📦 X-Eksen (Kapasite Attr)",
                        options=kap_x_cols,
                        format_func=lambda x: kap_x_labels.get(x, x),
                        key='kap_x_select'
                    )
                else:
                    kapasite_x_col = kap_x_cols[0]
                    st.caption(f"X → {kap_x_labels.get(kapasite_x_col, kapasite_x_col)}")

            # ── Filter uygula ────────────────────────────────────────────────
            filtered = (results if seçilen_kategori == '🔄 Tümü'
                        else results[results[urun_kategori_col] == seçilen_kategori])

            # ── KPI Metrics ──────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Satır",  f"{len(filtered):,}")
            c2.metric("Mağaza",        f"{filtered[urun_magaza_col].nunique():,}")
            c3.metric("Ürün",          f"{filtered[urun_urun_col].nunique():,}")
            c4.metric("Kombine Grup",  f"{filtered['Kombine_Grup'].nunique()}")
            c5.metric("Birleşik Grup", f"{filtered['Birlesik_Grup'].nunique()}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # 3D SCATTER
            # ══════════════════════════════════════════════════════════════════
            x_label = kap_x_labels.get(kapasite_x_col, kapasite_x_col)
            st.markdown(
                f"**3D Scatter** — X: {x_label} (Kapasite) &nbsp;|&nbsp; "
                f"Y: {urun_metric_col} (Ürün) &nbsp;|&nbsp; "
                f"Z: {urun_fiyat_col} (Fiyat)"
            )

            fig = px.scatter_3d(
                filtered,
                x=kapasite_x_col,
                y=urun_metric_col,
                z=urun_fiyat_col,
                color='Kombine_Grup',
                hover_data=[urun_magaza_col, urun_urun_col, urun_kategori_col,
                            'Kapasite_Grubu', 'Urun_Grubu', 'Fiyat_Grubu',
                            'Birlesik_Grup', 'Agirlikli_Skor'],
                opacity=0.78,
                height=540,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis_title=x_label,
                    yaxis_title=urun_metric_col,
                    zaxis_title=urun_fiyat_col,
                    xaxis=dict(backgroundcolor='#f0f4ff'),
                    yaxis=dict(backgroundcolor='#fff4f0'),
                    zaxis=dict(backgroundcolor='#f0fff4'),
                ),
                legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11))
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # BİRLEŞİK GRUP ÖZETİ — DETAYLI İSTATİSTİK
            # ══════════════════════════════════════════════════════════════════
            st.markdown("**📊 Birleşik Grup Özeti (Ağırlıklı)**")

            # Kapasite değişkenini belirle
            if len(kap_x_cols) > 1:
                filtered['_kap_avg'] = filtered[kap_x_cols].mean(axis=1)
                kap_stat_col = '_kap_avg'
                kap_stat_name = 'Kapasite(Ort)'
            else:
                kap_stat_col = kap_x_cols[0]
                kap_stat_name = kap_x_labels.get(kap_stat_col, kap_stat_col)

            group_order = ['TOP1', 'TOP2', 'TOP3', 'MID1', 'MID2', 'MID3',
                           'ALL1', 'ALL2', 'ALL3']

            summary_rows = []
            for grp in group_order:
                grp_data = filtered[filtered['Birlesik_Grup'] == grp]
                if len(grp_data) == 0:
                    continue
                n = len(grp_data)
                n_mag = grp_data[urun_magaza_col].nunique()

                kv = grp_data[kap_stat_col]
                uv = grp_data[urun_metric_col]
                fv = grp_data[urun_fiyat_col]
                sv = grp_data['Agirlikli_Skor']

                row = {
                    'Grup': grp,
                    'Satir': n,
                    'Magaza': n_mag,
                    f'{kap_stat_name}_Ort': round(kv.mean(), 2),
                    f'{kap_stat_name}_Min': round(kv.min(), 2),
                    f'{kap_stat_name}_Max': round(kv.max(), 2),
                    f'{kap_stat_name}_Std': round(kv.std(), 2) if n > 1 else 0,
                    f'{urun_metric_col}_Ort': round(uv.mean(), 2),
                    f'{urun_metric_col}_Min': round(uv.min(), 2),
                    f'{urun_metric_col}_Max': round(uv.max(), 2),
                    f'{urun_metric_col}_Std': round(uv.std(), 2) if n > 1 else 0,
                    f'{urun_fiyat_col}_Ort': round(fv.mean(), 2),
                    f'{urun_fiyat_col}_Min': round(fv.min(), 2),
                    f'{urun_fiyat_col}_Max': round(fv.max(), 2),
                    f'{urun_fiyat_col}_Std': round(fv.std(), 2) if n > 1 else 0,
                    'Skor_Ort': round(sv.mean(), 4),
                    'Skor_Std': round(sv.std(), 4) if n > 1 else 0,
                    'Varyans': round(sv.var(), 4) if n > 1 else 0,
                    'CV%': round(sv.std() / sv.mean() * 100, 1) if sv.mean() != 0 and n > 1 else 0,
                    'SE': round(sv.std() / np.sqrt(n), 4) if n > 1 else 0,
                }
                summary_rows.append(row)

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, hide_index=True, use_container_width=True,
                             height=380)

                # Birleşik Grup dağılım grafiği
                st.markdown("**📈 Birleşik Grup — Ağırlıklı Skor Dağılımı**")
                fig_unified = px.box(
                    filtered, x='Birlesik_Grup', y='Agirlikli_Skor',
                    color='Birlesik_Grup', height=320,
                    category_orders={'Birlesik_Grup': group_order},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_unified.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title='Birleşik Grup',
                    yaxis_title='Ağırlıklı Skor'
                )
                st.plotly_chart(fig_unified, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # PER-KATEGORI DAĞILIM (tek kategori seçilince)
            # ══════════════════════════════════════════════════════════════════
            if seçilen_kategori != '🔄 Tümü':
                st.markdown(f"**📈 Dağılım — {seçilen_kategori}**")

                col_y, col_z = st.columns(2)

                with col_y:
                    st.caption(f"Ürün Metrik: {urun_metric_col}")
                    fig_y = px.box(
                        filtered, x='Urun_Grubu', y=urun_metric_col,
                        color='Urun_Grubu', height=270,
                        color_discrete_sequence=['#FFC7CE', '#FFEB9C', '#C6EFCE',
                                                 '#BDD7EE', '#E2EFDA', '#FCE4D6']
                    )
                    fig_y.update_layout(showlegend=False,
                                        margin=dict(l=0, r=0, t=10, b=0),
                                        xaxis_title='Ürün Grubu')
                    st.plotly_chart(fig_y, use_container_width=True)

                with col_z:
                    st.caption(f"Fiyat: {urun_fiyat_col}")
                    fig_z = px.box(
                        filtered, x='Fiyat_Grubu', y=urun_fiyat_col,
                        color='Fiyat_Grubu', height=270,
                        color_discrete_sequence=['#BDD7EE', '#E2EFDA', '#FCE4D6',
                                                 '#FFC7CE', '#FFEB9C', '#C6EFCE']
                    )
                    fig_z.update_layout(showlegend=False,
                                        margin=dict(l=0, r=0, t=10, b=0),
                                        xaxis_title='Fiyat Grubu')
                    st.plotly_chart(fig_z, use_container_width=True)

                # İstatistik tabloları
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    stats_y = (filtered.groupby('Urun_Grubu')[urun_metric_col]
                               .agg(['count', 'mean', 'min', 'max']).round(2))
                    stats_y.columns = ['Adet', 'Ort', 'Min', 'Max']
                    st.dataframe(stats_y, use_container_width=True)
                with col_s2:
                    stats_z = (filtered.groupby('Fiyat_Grubu')[urun_fiyat_col]
                               .agg(['count', 'mean', 'min', 'max']).round(2))
                    stats_z.columns = ['Adet', 'Ort', 'Min', 'Max']
                    st.dataframe(stats_z, use_container_width=True)

            else:
                # Tümü seçilince: Kategori özet tablo
                st.markdown("**📈 Kategori Bazında Özet**")
                kat_summary = (filtered
                               .groupby(urun_kategori_col)
                               .agg(
                                   Satır=(urun_urun_col, 'count'),
                                   Mağaza=(urun_magaza_col, 'nunique'),
                                   Ürün=(urun_urun_col, 'nunique'),
                                   Kombine_Grup=('Kombine_Grup', 'nunique')
                               )
                               .reset_index())
                st.dataframe(kat_summary, hide_index=True, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # İNDİRME
            # ══════════════════════════════════════════════════════════════════
            st.markdown("**📥 İndir**")

            # Çıktı sütunları — internal _Kap_X_ colonlar dahil edilmez
            show_cols = list(dict.fromkeys([
                urun_magaza_col, urun_urun_col, urun_kategori_col,
                'Kapasite_Grubu', urun_metric_col, 'Urun_Grubu',
                urun_fiyat_col, 'Fiyat_Grubu', 'Kombine_Grup',
                'Agirlikli_Skor', 'Birlesik_Grup'
            ]))

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results[show_cols].to_excel(writer, index=False, sheet_name='Tüm Sonuç')
                    for kat in kategoriler:
                        kat_df     = results[results[urun_kategori_col] == kat][show_cols]
                        sheet_name = str(kat)[:31].replace('/', '-').replace('\\', '-')
                        kat_df.to_excel(writer, index=False, sheet_name=sheet_name)

                st.download_button(
                    "📥 Excel (Kategori Sheeti)",
                    buffer.getvalue(),
                    f"cluster_3d_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )

            with col_d2:
                csv_data = results[show_cols].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 CSV",
                    csv_data,
                    f"cluster_3d_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    use_container_width=True
                )

            # ── Detaylı Tablo ────────────────────────────────────────────────
            with st.expander("📋 Tüm Veriyi Göster", expanded=False):
                st.dataframe(filtered[show_cols], height=420, use_container_width=True)

        else:
            # Demo 3D animasyon + yönlendirme
            st.markdown("""
            <div class="empty-state-wrap">
                <div class="empty-title">📈 Demo — Örnek 3D Cluster Görünümü</div>
                <div class="empty-sub">Sol panelden veri yüklediğinizde gerçek analiz başlayacak</div>
            </div>
            """, unsafe_allow_html=True)
            render_demo_3d()


if __name__ == "__main__":
    main()
