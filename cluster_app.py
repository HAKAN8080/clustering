"""
Cluster Analizi — Mağaza Kapasite + Ürün (K-Means)
Format: 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3

Metodoloji:
1. Kapasite (K-Means): Mağazaları global olarak 1/2/3 gruplarına ayır
2. Ürün (K-Means): Her kategori için mağazaları performansa göre 1/2/3 gruplarına ayır
3. Kombine: Kapasite Grubu + Ürün Grubu → 1-1, 1-2, ... 3-3
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
    .hdr-badge.g1 { background: rgba(74,222,128,0.2); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .hdr-badge.g2 { background: rgba(250,204,21,0.2); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
    .hdr-badge.g3 { background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }

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
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None


def kmeans_global(df, attribute_cols, n_clusters, desc=True):
    """
    Global K-Means — kapasite gruplama (tüm mağazalar üzerinde bir kez).
    desc=True → 1=en büyük (TOP), desc=False → 1=en küçük (ALL)
    """
    X = df[attribute_cols].copy()

    # NaN değerleri ortalama ile doldur
    for col in X.columns:
        col_mean = X[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        X[col] = X[col].fillna(col_mean)

    # inf değerleri temizle
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Yeterli veri kontrolü
    if len(X) < n_clusters:
        return np.ones(len(X), dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # NaN kontrolü (StandardScaler sonrası)
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    means = {c: X.iloc[clusters == c].mean().mean() for c in range(n_clusters)}
    sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def kmeans_store_category(df, magaza_col, kategori_col, metric_cols, n_clusters=3, desc=True, min_point=30):
    """
    Mağaza + Kategori bazında ürün gruplaması (ESKİ - kategori bazlı).
    Artık kullanılmıyor, geriye dönük uyumluluk için tutuldu.
    """
    # ADIM 1: Mağaza + Kategori bazında aggregate + point sayısı
    store_cat_df = df.groupby([magaza_col, kategori_col], as_index=False).agg(
        **{col: (col, 'sum') for col in metric_cols},
        _point_count=(metric_cols[0], 'count')
    )

    store_cat_df['_total_metric'] = store_cat_df[metric_cols].sum(axis=1)
    store_cat_df['_low_point'] = store_cat_df['_point_count'] < min_point
    store_cat_df['Urun_Grubu'] = 1

    return store_cat_df[[magaza_col, kategori_col, 'Urun_Grubu', '_total_metric']]


def calculate_weighted_score(df, metric_cols, weights):
    """
    Global Weighted Score hesapla (CoPilot yaklaşımı).

    1. Her metriği Min-Max normalize et (0-1 arası)
    2. Ağırlıklı toplam hesapla

    Args:
        df: DataFrame
        metric_cols: Liste - normalize edilecek metrik kolonları
        weights: Dict - {kolon_adı: ağırlık} şeklinde ağırlıklar

    Returns:
        Series: Weighted score değerleri
    """
    scores = pd.DataFrame(index=df.index)

    for col in metric_cols:
        if col not in df.columns:
            continue

        values = df[col].copy()

        # NaN ve inf temizle
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.fillna(0)
        values = values.clip(lower=0)  # Negatif değerleri 0 yap

        # Min-Max Normalization (0-1 arası)
        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            scores[col] = (values - min_val) / (max_val - min_val)
        else:
            scores[col] = 0.5  # Tüm değerler aynıysa 0.5 ata

    # Ağırlıklı toplam hesapla
    weighted_score = pd.Series(0.0, index=df.index)
    total_weight = 0

    for col in metric_cols:
        if col in scores.columns and col in weights:
            weighted_score += scores[col] * weights[col]
            total_weight += weights[col]

    # Normalize et (toplam ağırlık 1 değilse)
    if total_weight > 0:
        weighted_score = weighted_score / total_weight

    return weighted_score


def kmeans_global_weighted(df, magaza_col, kategori_col, metric_cols, weights, n_clusters=3, desc=True):
    """
    Global Weighted Score tabanlı performans gruplaması.

    1. Mağaza + Kategori bazında metrikleri topla
    2. Weighted Score hesapla
    3. Global K-Means ile performans grubu ata

    desc=True → yüksek skor = 1 (Hızlı/TOP performans)
    """
    # ADIM 1: Mağaza + Kategori bazında aggregate
    store_cat_df = df.groupby([magaza_col, kategori_col], as_index=False).agg(
        **{col: (col, 'sum') for col in metric_cols},
        _point_count=(metric_cols[0], 'count')
    )

    # ADIM 2: Weighted Score hesapla
    store_cat_df['_weighted_score'] = calculate_weighted_score(
        store_cat_df, metric_cols, weights
    )

    # ADIM 3: Global K-Means
    X = store_cat_df['_weighted_score'].values.reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Yeterli veri kontrolü
    n_unique = len(np.unique(X))
    actual_clusters = min(n_clusters, n_unique, len(X))

    if actual_clusters < 2:
        store_cat_df['Urun_Grubu'] = 1
    else:
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Sıralama: desc=True → yüksek skor = 1 (Hızlı)
        means = {c: X[clusters == c].mean() for c in range(actual_clusters)}
        sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
        mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

        store_cat_df['Urun_Grubu'] = [mapping[c] for c in clusters]

    return store_cat_df[[magaza_col, kategori_col, 'Urun_Grubu', '_weighted_score']]


def get_kapasite_label(grup_num, total):
    """Kapasite grup numarası → 1, 2, 3 (sayısal)"""
    return str(grup_num)


def assign_experiential_cluster(values, n_clusters=3, desc=True):
    """
    Experiential Scoring için tek bir ekseni n_clusters kümeye böl.
    Quantile tabanlı bölme yapar (eşit sayıda eleman her grupta).
    desc=True → yüksek değer = küme 1 (TOP)
    desc=False → düşük değer = küme 1
    Döndürür: 1, 2, 3, ... şeklinde küme numaraları
    """
    values = pd.Series(values).copy()

    # inf değerleri NaN yap
    values = values.replace([np.inf, -np.inf], np.nan)

    # NaN'ları ortalama ile doldur
    fill_value = values.mean()
    if pd.isna(fill_value):
        fill_value = 0
    values = values.fillna(fill_value)

    # Boş veri kontrolü
    if len(values) == 0:
        return np.array([1])

    # Quantile tabanlı kümeleme
    try:
        clusters = pd.qcut(values, q=n_clusters, labels=False, duplicates='drop')
        actual_clusters = clusters.max() + 1
        if pd.isna(actual_clusters):
            actual_clusters = 1
            clusters = pd.Series([0] * len(values))
    except (ValueError, TypeError):
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
            <div class="splash-sub">K-Means Clustering</div>
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

    labels = (['1-1'] * n) + (['2-2'] * n) + (['3-3'] * n)
    xs = np.concatenate([c1[:, 0], c2[:, 0], c3[:, 0]])
    ys = np.concatenate([c1[:, 1], c2[:, 1], c3[:, 1]])
    zs = np.concatenate([c1[:, 2], c2[:, 2], c3[:, 2]])

    import plotly.graph_objects as go

    colors = {'1-1': '#60a5fa', '2-2': '#4ade80', '3-3': '#f472b6'}
    fig = go.Figure()
    for label in ['1-1', '2-2', '3-3']:
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

    # Grup isimleri için default değerler
    if 'grup_isimleri' not in st.session_state:
        st.session_state.grup_isimleri = {
            '1-1': 'TOP 1', '1-2': 'TOP 2', '1-3': 'TOP 3',
            '2-1': 'MID 1', '2-2': 'MID 2', '2-3': 'MID 3',
            '3-1': 'ALL 1', '3-2': 'ALL 2', '3-3': 'ALL 3'
        }

    # ── Ana Başlık ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-main-header">
        <h1>📊 3D <span>Cluster</span> Analizi</h1>
        <div class="header-sub">Mağaza · Kategori · Performans &nbsp;|&nbsp; K-Means Clustering</div>
        <div class="header-badge-row">
            <span class="hdr-badge g1">1-1</span>
            <span class="hdr-badge g2">2-2</span>
            <span class="hdr-badge g3">3-3</span>
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
                loaded = load_data(uploaded_kap)
                if loaded is not None:
                    st.session_state.kapasite_df = loaded
                    st.session_state._kap_file_id = file_id
                else:
                    st.session_state.kapasite_df = None

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
            # "capacity" içeren kolonları otomatik seç
            capacity_cols = [c for c in available_k if 'capacity' in c.lower()]
            default_cols = capacity_cols if capacity_cols else available_k[:min(2, len(available_k))]

            kap_attrs   = st.multiselect(
                "📊 Kapasite Attributeları (X-eksen)",
                options=available_k,
                default=default_cols,
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
                loaded = load_data(uploaded_urun)
                if loaded is not None:
                    st.session_state.urun_df = loaded
                    st.session_state._urun_file_id = file_id
                else:
                    st.session_state.urun_df = None

        # Defaults
        urun_magaza_col   = None
        urun_kategori_col = None
        urun_metric_col   = None
        urun_fiyat_col    = None
        urun_third_col    = None
        urun_grup_sayisi  = 3
        fiyat_grup_sayisi = 3
        w1, w2, w3        = 40, 40, 20  # Default ağırlıklar

        if st.session_state.urun_df is not None:
            df_u           = st.session_state.urun_df
            all_cols_u     = df_u.columns.tolist()
            numeric_cols_u = df_u.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols_u     = df_u.select_dtypes(include=['object', 'category']).columns.tolist()

            # Mağaza kolonu
            urun_magaza_col = st.selectbox("🏪 Mağaza Kolonu", options=all_cols_u, key='u_magaza')

            # Kategori kolonu
            cat_options = [c for c in cat_cols_u if c != urun_magaza_col]
            if not cat_options:
                cat_options = [c for c in all_cols_u
                               if c != urun_magaza_col and c not in numeric_cols_u]

            if cat_options:
                urun_kategori_col = st.selectbox(
                    "🏷️ Kategori Kolonu", options=cat_options, key='u_kategori'
                )
            else:
                st.warning("⚠️ Kategori (string) kolonu bulunamadı.")

            # Kullanılan kolonlar
            used = [urun_magaza_col]
            if urun_kategori_col:
                used.append(urun_kategori_col)
            metric_options = [c for c in numeric_cols_u if c not in used]

            # ═══════════════════════════════════════════════════════════════
            # METRİK SEÇİMİ
            # ═══════════════════════════════════════════════════════════════
            if len(metric_options) >= 2:
                urun_metric_col = st.selectbox(
                    "📈 Ürün Metrik (Y-eksen)", options=metric_options, key='u_metric'
                )
                fiyat_options = [c for c in metric_options if c != urun_metric_col]
                urun_fiyat_col = st.selectbox(
                    "💰 İkinci Metrik (Z-eksen)", options=fiyat_options, key='u_fiyat'
                )

                # Üçüncü metrik seçimi (opsiyonel)
                third_options = [c for c in metric_options if c != urun_metric_col and c != urun_fiyat_col]
                if third_options:
                    urun_third_col = st.selectbox(
                        "📊 Üçüncü Metrik (Opsiyonel)",
                        options=['-- Seçme --'] + third_options,
                        key='u_third'
                    )
                    if urun_third_col == '-- Seçme --':
                        urun_third_col = None
                else:
                    urun_third_col = None

                col_grp, col_pnt = st.columns(2)
                with col_grp:
                    urun_grup_sayisi = st.number_input(
                        "Ürün Grup Sayısı", min_value=2, max_value=10, value=3, key='u_grup'
                    )
                with col_pnt:
                    min_point = st.number_input(
                        "Min Point (Mağaza-Kat)", min_value=1, max_value=1000, value=1, key='min_point',
                        help="Mağaza-Kategori kombinasyonunda en az kaç satır (point) olmalı"
                    )

                # ═══════════════════════════════════════════════════════════════
                # AĞIRLIK AYARLARI (Weighted Score için)
                # ═══════════════════════════════════════════════════════════════
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">⚖️ Performans Ağırlıkları</div>',
                            unsafe_allow_html=True)
                st.caption("Weighted Score hesaplaması için metrik ağırlıkları (toplam: 100)")

                # Ağırlık slider'ları
                w1 = st.slider(
                    f"📈 {urun_metric_col[:20]}...",
                    min_value=0, max_value=100, value=40,
                    key='weight_metric1',
                    help="Birinci metriğin ağırlığı"
                )
                w2 = st.slider(
                    f"💰 {urun_fiyat_col[:20]}...",
                    min_value=0, max_value=100, value=40,
                    key='weight_metric2',
                    help="İkinci metriğin ağırlığı"
                )
                if urun_third_col:
                    w3 = st.slider(
                        f"📊 {urun_third_col[:20]}...",
                        min_value=0, max_value=100, value=20,
                        key='weight_metric3',
                        help="Üçüncü metriğin ağırlığı"
                    )
                else:
                    w3 = 0

                total_weight = w1 + w2 + w3
                if total_weight > 0:
                    st.caption(f"✓ Toplam ağırlık: {total_weight} → Normalize edilecek")
                else:
                    st.warning("⚠️ En az bir ağırlık > 0 olmalı!")

            elif len(metric_options) == 1:
                urun_metric_col = metric_options[0]
                urun_third_col = None
                min_point = 1
                w1, w2, w3 = 100, 0, 0
                st.warning("⚠️ İkinci metrik kolonu için yeterli sayısal kolon yok.")
            else:
                urun_third_col = None
                min_point = 1
                w1, w2, w3 = 34, 33, 33
                st.warning("⚠️ Sayısal kolon bulunamadı.")

            n_kat = df_u[urun_kategori_col].nunique() if urun_kategori_col else 0
            n_mag = df_u[urun_magaza_col].nunique() if urun_magaza_col else 0
            st.caption(f"✓ {len(df_u)} satır | {n_mag} mağaza | {n_kat} kategori")
        else:
            st.caption("📁 ürün_data.xlsx yükleyin")

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

        # ─── GRUPLA BUTONU ───────────────────────────────────────────────────
        btn_disabled = (
            st.session_state.kapasite_df is None
            or st.session_state.urun_df is None
            or len(kap_attrs) == 0
            or urun_metric_col is None
            or urun_kategori_col is None
        )

        if st.button("🚀 Grupla ve Birleştir",
                      disabled=btn_disabled, use_container_width=True, type="primary"):
            with st.spinner("Gruplama yapılıyor…"):
              try:
                # ── STEP 1: Kapasite gruplama (GLOBAL K-Means) ──────────────
                kap_df_raw = st.session_state.kapasite_df.copy()

                # ⚡ MAĞAZA BAZINDA UNIQUE KAPASITE: Aynı mağazanın tekrar eden
                # satırlarını birleştir (max değeri al)
                agg_dict = {col: 'max' for col in kap_attrs}
                kap_df = kap_df_raw.groupby(kap_label, as_index=False).agg(agg_dict)

                st.info(f"📊 {len(kap_df_raw)} satırdan {len(kap_df)} unique mağaza oluşturuldu")

                kap_df['_Kap_Grup_Num'] = kmeans_global(
                    kap_df, kap_attrs, kap_grup_sayisi, desc=desc_order
                )
                kap_df['Kapasite_Grubu'] = kap_df['_Kap_Grup_Num'].apply(
                    lambda x: get_kapasite_label(x, kap_grup_sayisi)
                )
                st.session_state.kapasite_results = kap_df

                # ── STEP 2: Ürün df hazırla + VERİ TEMİZLİĞİ ─────────────────
                urun_df = st.session_state.urun_df.copy()

                # 🧹 VERİ TEMİZLİĞİ: Metrik kolonlarında NaN → 0, Negatif → 0
                metric_cols_to_clean = [urun_metric_col]
                if urun_fiyat_col:
                    metric_cols_to_clean.append(urun_fiyat_col)

                for col in metric_cols_to_clean:
                    if col in urun_df.columns:
                        # NaN → 0
                        urun_df[col] = urun_df[col].fillna(0)
                        # Negatif → 0
                        urun_df[col] = urun_df[col].clip(lower=0)

                # 📊 POINT FİLTRESİ: Mağaza-Kategori kombinasyonu için min satır sayısı
                point_counts = urun_df.groupby([urun_magaza_col, urun_kategori_col]).size().reset_index(name='_point_count')
                urun_df = urun_df.merge(point_counts, on=[urun_magaza_col, urun_kategori_col], how='left')

                # Point eşiğinin altındaki kombinasyonları işaretle
                low_point_mask = urun_df['_point_count'] < min_point
                low_point_count = urun_df[low_point_mask][urun_magaza_col + '_' + urun_kategori_col if False else urun_magaza_col].nunique() if low_point_mask.any() else 0

                excluded_combinations = urun_df[low_point_mask][[urun_magaza_col, urun_kategori_col]].drop_duplicates()
                n_excluded = len(excluded_combinations)

                if n_excluded > 0:
                    st.warning(f"⚠️ {n_excluded} mağaza-kategori kombinasyonu {min_point} point altında (gruplamaya dahil edilmeyecek)")

                # ── STEP 3: Kapasite join ────────────────────────────────────
                join_cols   = [kap_label, 'Kapasite_Grubu'] + kap_attrs
                kap_join    = kap_df[join_cols].copy()
                rename_map  = {col: f'_Kap_X_{col}' for col in kap_attrs}
                kap_join    = kap_join.rename(columns={**rename_map, kap_label: urun_magaza_col})

                urun_df = urun_df.merge(kap_join, on=urun_magaza_col, how='left')
                urun_df['Kapasite_Grubu'] = urun_df['Kapasite_Grubu'].fillna('?')

                unmatched = urun_df['Kapasite_Grubu'].eq('?').sum()

                # ═══════════════════════════════════════════════════════════════
                # K-MEANS CLUSTERING (Global Weighted Score Tabanlı)
                # ═══════════════════════════════════════════════════════════════
                # Metrik kolonları ve ağırlıkları hazırla
                metric_cols_for_grouping = [urun_metric_col]
                weights = {urun_metric_col: w1}

                if urun_fiyat_col:
                    metric_cols_for_grouping.append(urun_fiyat_col)
                    weights[urun_fiyat_col] = w2

                if urun_third_col:
                    metric_cols_for_grouping.append(urun_third_col)
                    weights[urun_third_col] = w3

                st.info(f"📊 Weighted Score: {', '.join([f'{k[:15]}={v}%' for k,v in weights.items()])}")

                # Global Weighted Score tabanlı performans gruplaması
                store_cat_groups = kmeans_global_weighted(
                    urun_df, urun_magaza_col, urun_kategori_col,
                    metric_cols_for_grouping, weights,
                    n_clusters=urun_grup_sayisi, desc=desc_order
                )

                # Ana veriye join et
                urun_df = urun_df.merge(
                    store_cat_groups[[urun_magaza_col, urun_kategori_col, 'Urun_Grubu']],
                    on=[urun_magaza_col, urun_kategori_col],
                    how='left'
                )
                urun_df['Urun_Grubu'] = urun_df['Urun_Grubu'].fillna(1).astype(int)

                # Kombine Grup → 1-1, 1-2, ... 3-3 format
                urun_df['Kombine_Grup'] = (
                    urun_df['Kapasite_Grubu'].astype(str) + '-' +
                    urun_df['Urun_Grubu'].astype(str)
                )

                # Weighted Score'u da ana veriye ekle
                urun_df = urun_df.merge(
                    store_cat_groups[[urun_magaza_col, urun_kategori_col, '_weighted_score']],
                    on=[urun_magaza_col, urun_kategori_col],
                    how='left'
                )
                urun_df['_weighted_score'] = urun_df['_weighted_score'].fillna(0)

                # Session config
                st.session_state.final_results = urun_df
                st.session_state.config = {
                    'urun_magaza_col':   urun_magaza_col,
                    'urun_kategori_col': urun_kategori_col,
                    'urun_metric_col':   urun_metric_col,
                    'urun_fiyat_col':    urun_fiyat_col,
                    'urun_third_col':    urun_third_col,
                    'kap_label':         kap_label,
                    'kap_attrs':         kap_attrs,
                    'kap_x_cols':        [f'_Kap_X_{c}' for c in kap_attrs],
                    'kap_x_labels':      {f'_Kap_X_{c}': c for c in kap_attrs},
                    'unmatched':         unmatched,
                    'weights':           weights,
                }

                if unmatched > 0:
                    st.warning(f"⚠️ {unmatched} satırda mağaza eşleştirme yapılamadı — '?' olarak bırakıldı.")
                else:
                    st.success("✅ Gruplama tamamlandı!")

              except Exception as e:
                st.error(f"❌ Hesaplama hatası: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="text")

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
            urun_kategori_col = cfg['urun_kategori_col']
            kap_x_cols        = cfg['kap_x_cols']
            kap_x_labels      = cfg['kap_x_labels']

            # ── Legenda ──────────────────────────────────────────────────────
            st.markdown("""
            <div class="legend-box">
                <b>Format → 1-1, 1-2, ... 3-3</b><br>
                <b>Kapasite Grubu:</b> 1 / 2 / 3 — Mağaza bazlı (global K-Means)<br>
                <b>Ürün Grubu:</b> 1 / 2 / 3 — Mağaza+Kategori bazlı K-Means
            </div>
            """, unsafe_allow_html=True)

            # ── Elle Grup İsimlendirme ────────────────────────────────────────
            with st.expander("✏️ Grup İsimlerini Düzenle", expanded=False):
                st.caption("Her gruba özel isim verin (Excel'de bu isimler kullanılacak)")

                # Mevcut grupları al
                existing_groups = sorted(results['Kombine_Grup'].unique())
                group_order = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
                ordered_groups = [g for g in group_order if g in existing_groups]
                for g in existing_groups:
                    if g not in ordered_groups:
                        ordered_groups.append(g)

                # 3 sütunlu layout
                cols = st.columns(3)
                for i, grup in enumerate(ordered_groups):
                    col_idx = i % 3
                    with cols[col_idx]:
                        default_name = st.session_state.grup_isimleri.get(grup, grup)
                        new_name = st.text_input(
                            f"Grup {grup}",
                            value=default_name,
                            key=f'grup_isim_{grup}',
                            label_visibility="visible"
                        )
                        st.session_state.grup_isimleri[grup] = new_name

                # Grup İsmi kolonunu ekle
                results['Grup_Ismi'] = results['Kombine_Grup'].map(st.session_state.grup_isimleri)
                st.session_state.final_results['Grup_Ismi'] = st.session_state.final_results['Kombine_Grup'].map(st.session_state.grup_isimleri)

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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Satır",  f"{len(filtered):,}")
            c2.metric("Mağaza",        f"{filtered[urun_magaza_col].nunique():,}")
            c3.metric("Kategori",      f"{filtered[urun_kategori_col].nunique():,}")
            c4.metric("Kombine Grup",  f"{filtered['Kombine_Grup'].nunique()}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # 3D SCATTER
            # ══════════════════════════════════════════════════════════════════
            x_label = kap_x_labels.get(kapasite_x_col, kapasite_x_col)
            urun_metric_col = cfg.get('urun_metric_col')
            urun_fiyat_col = cfg.get('urun_fiyat_col')

            if urun_metric_col and urun_fiyat_col:
                st.markdown(
                    f"**3D Scatter** — X: {x_label} (Kapasite) &nbsp;|&nbsp; "
                    f"Y: {urun_metric_col} &nbsp;|&nbsp; "
                    f"Z: {urun_fiyat_col}"
                )

                fig = px.scatter_3d(
                    filtered,
                    x=kapasite_x_col,
                    y=urun_metric_col,
                    z=urun_fiyat_col,
                    color='Kombine_Grup',
                    hover_data=[urun_magaza_col, urun_kategori_col,
                                'Kapasite_Grubu', 'Urun_Grubu'],
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
            # KOMBİNE GRUP ÖZETİ — DETAYLI İSTATİSTİK
            # ══════════════════════════════════════════════════════════════════
            st.markdown("**📊 Kombine Grup Özeti**")

            # Kapasite değişkenini belirle
            if len(kap_x_cols) > 1:
                filtered['_kap_avg'] = filtered[kap_x_cols].mean(axis=1)
                kap_stat_col = '_kap_avg'
                kap_stat_name = 'Kapasite(Ort)'
            else:
                kap_stat_col = kap_x_cols[0]
                kap_stat_name = kap_x_labels.get(kap_stat_col, kap_stat_col)

            # Grup sırası: 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3
            group_order = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
            all_groups = [g for g in group_order if g in filtered['Kombine_Grup'].values]
            # Eğer farklı gruplar varsa ekle
            for g in sorted(filtered['Kombine_Grup'].unique()):
                if g not in all_groups:
                    all_groups.append(g)

            summary_rows = []
            for grp in all_groups:
                grp_data = filtered[filtered['Kombine_Grup'] == grp]
                if len(grp_data) == 0:
                    continue
                n = len(grp_data)
                n_mag = grp_data[urun_magaza_col].nunique()

                kv = grp_data[kap_stat_col]

                # Grup ismini al
                grup_ismi = st.session_state.grup_isimleri.get(grp, grp)

                row = {
                    'Grup': grp,
                    'Grup İsmi': grup_ismi,
                    'Satır': n,
                    'Mağaza': n_mag,
                    f'{kap_stat_name}_Ort': round(kv.mean(), 2),
                }

                urun_metric_col = cfg.get('urun_metric_col')
                urun_fiyat_col = cfg.get('urun_fiyat_col')
                if urun_metric_col and urun_metric_col in grp_data.columns:
                    row[f'{urun_metric_col[:10]}_Ort'] = round(grp_data[urun_metric_col].mean(), 2)
                if urun_fiyat_col and urun_fiyat_col in grp_data.columns:
                    row[f'{urun_fiyat_col[:10]}_Ort'] = round(grp_data[urun_fiyat_col].mean(), 2)

                # Weighted Score ortalaması
                if '_weighted_score' in grp_data.columns:
                    row['W.Score_Ort'] = round(grp_data['_weighted_score'].mean(), 4)

                summary_rows.append(row)

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, hide_index=True, use_container_width=True,
                             height=380)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # PER-KATEGORI DAĞILIM (tek kategori seçilince)
            # ══════════════════════════════════════════════════════════════════
            if seçilen_kategori != '🔄 Tümü':
                st.markdown(f"**📈 Dağılımlar — {seçilen_kategori}**")

                urun_metric_col = cfg.get('urun_metric_col')
                urun_fiyat_col = cfg.get('urun_fiyat_col')

                col_y, col_z = st.columns(2)
                with col_y:
                    if urun_metric_col and urun_metric_col in filtered.columns:
                        st.caption(f"Ürün Metrik: {urun_metric_col}")
                        fig_y = px.box(
                            filtered, x='Urun_Grubu', y=urun_metric_col,
                            color='Urun_Grubu', height=270,
                            color_discrete_sequence=['#C6EFCE', '#FFEB9C', '#FFC7CE']
                        )
                        fig_y.update_layout(showlegend=False,
                                            margin=dict(l=0, r=0, t=10, b=0),
                                            xaxis_title='Ürün Grubu')
                        st.plotly_chart(fig_y, use_container_width=True)
                with col_z:
                    if urun_fiyat_col and urun_fiyat_col in filtered.columns:
                        st.caption(f"İkinci Metrik: {urun_fiyat_col}")
                        fig_z = px.box(
                            filtered, x='Urun_Grubu', y=urun_fiyat_col,
                            color='Urun_Grubu', height=270,
                            color_discrete_sequence=['#BDD7EE', '#E2EFDA', '#FCE4D6']
                        )
                        fig_z.update_layout(showlegend=False,
                                            margin=dict(l=0, r=0, t=10, b=0),
                                            xaxis_title='Ürün Grubu')
                        st.plotly_chart(fig_z, use_container_width=True)

            else:
                # Tümü seçilince: Kategori özet tablo
                st.markdown("**📈 Kategori Bazında Özet**")
                kat_summary = (filtered
                               .groupby(urun_kategori_col)
                               .agg(
                                   Satır=(urun_magaza_col, 'count'),
                                   Mağaza=(urun_magaza_col, 'nunique'),
                                   Kombine_Grup=('Kombine_Grup', 'nunique')
                               )
                               .reset_index())
                st.dataframe(kat_summary, hide_index=True, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # İNDİRME
            # ══════════════════════════════════════════════════════════════════
            st.markdown("**📥 İndir**")

            # Çıktı sütunları
            show_cols = [urun_magaza_col, urun_kategori_col, 'Kapasite_Grubu']

            urun_metric_col = cfg.get('urun_metric_col')
            urun_fiyat_col = cfg.get('urun_fiyat_col')
            if urun_metric_col:
                show_cols.append(urun_metric_col)
            if urun_fiyat_col:
                show_cols.append(urun_fiyat_col)
            show_cols.append('Urun_Grubu')
            show_cols.append('Kombine_Grup')
            show_cols.append('Grup_Ismi')

            show_cols = list(dict.fromkeys(show_cols))  # Remove duplicates
            show_cols = [c for c in show_cols if c in results.columns]  # Only existing columns

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
