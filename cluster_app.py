"""
Unified Cluster App — Model Seçimli
1. Thorius Algorithm: Kapasite + Ürün (Global K-Means)
2. CoPilot V3: LivingArea + MainGroup + SubGroup (Weighted K-Means + Min-10)
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime

st.set_page_config(page_title="Cluster Analizi", page_icon="📊", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# ORTAK SABİTLER & CSS
# ═══════════════════════════════════════════════════════════════════════════════

# CoPilot V3 Sabitleri - Yeni İsimlendirme
# TOP = Hızlı performans, MID = Orta performans, ALL = Yavaş performans
COPILOT_RANK_TO_CODE = {
    1: "TOP1", 2: "TOP2", 3: "TOP3",
    4: "MID4", 5: "MID5", 6: "MID6",
    7: "ALL7", 8: "ALL8", 9: "ALL9",
}
COPILOT_CODE_TO_RANK = {v: k for k, v in COPILOT_RANK_TO_CODE.items()}
COPILOT_RANK_TO_PAIR = {
    1: ("Büyük", "Hızlı"),  2: ("Orta", "Hızlı"),  3: ("Büyük", "Orta"),
    4: ("Küçük", "Hızlı"),  5: ("Orta", "Orta"),   6: ("Küçük", "Orta"),
    7: ("Büyük", "Yavaş"),  8: ("Orta", "Yavaş"),  9: ("Küçük", "Yavaş"),
}
COPILOT_KEY_TO_DETAIL = {
    "BüyükHızlı": "TOP1-Büyük-Hızlı", "OrtaHızlı": "TOP2-Orta-Hızlı",   "BüyükOrta": "TOP3-Büyük-Orta",
    "KüçükHızlı": "MID4-Küçük-Hızlı", "OrtaOrta": "MID5-Orta-Orta",     "KüçükOrta": "MID6-Küçük-Orta",
    "BüyükYavaş": "ALL7-Büyük-Yavaş", "OrtaYavaş": "ALL8-Orta-Yavaş",   "KüçükYavaş": "ALL9-Küçük-Yavaş",
}

# Thorius Sabitleri - Yeni İsimlendirme (Kapasite-Performans → Kod)
# Kapasite: 1=Büyük, 2=Orta, 3=Küçük | Performans: 1=Hızlı, 2=Orta, 3=Yavaş
THORIUS_KOMBINE_TO_CODE = {
    "1-1": "TOP1-Büyük-Hızlı",  "2-1": "TOP2-Orta-Hızlı",  "1-2": "TOP3-Büyük-Orta",
    "3-1": "MID4-Küçük-Hızlı",  "2-2": "MID5-Orta-Orta",   "3-2": "MID6-Küçük-Orta",
    "1-3": "ALL7-Büyük-Yavaş",  "2-3": "ALL8-Orta-Yavaş",  "3-3": "ALL9-Küçük-Yavaş",
}
THORIUS_KOMBINE_TO_SHORT = {
    "1-1": "TOP1", "2-1": "TOP2", "1-2": "TOP3",
    "3-1": "MID4", "2-2": "MID5", "3-2": "MID6",
    "1-3": "ALL7", "2-3": "ALL8", "3-3": "ALL9",
}

st.markdown("""
<style>
    /* Splash overlay */
    .splash-overlay {
        position: fixed; inset: 0; z-index: 9999;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        animation: splashFadeOut 0.6s ease 2.5s forwards;
        pointer-events: none;
    }
    @keyframes splashFadeOut {
        0% { opacity: 1; }
        100% { opacity: 0; pointer-events: none; }
    }
    .splash-title {
        font-size: 2.2rem; font-weight: 700;
        color: #fff; z-index: 2;
        animation: titlePop 0.7s ease 0.3s both;
    }
    .splash-title span { color: #60a5fa; }
    @keyframes titlePop {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Headers */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #2F5496 100%);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 4px 24px rgba(15,23,42,0.25);
    }
    .app-header h1 {
        font-size: 1.65rem; font-weight: 700;
        color: #fff; margin: 0;
    }
    .app-header h1 span { color: #60a5fa; }
    .app-header .sub {
        font-size: 0.8rem; color: #94a3b8;
        margin-top: 4px; letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .section-header {
        background: #C00000;
        color: white;
        padding: 9px 14px;
        font-weight: bold;
        font-size: 13px;
        margin: 8px 0;
        border-radius: 5px;
    }
    .section-header-blue {
        background: #2F5496;
        color: white;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 14px;
        margin: 8px 0;
        border-radius: 5px;
    }
    .section-header-green {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 13px;
        margin: 8px 0;
        border-radius: 5px;
    }
    .section-header-orange {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 13px;
        margin: 8px 0;
        border-radius: 5px;
    }

    .model-card {
        background: linear-gradient(135deg, #1e3a5f, #2F5496);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(47,84,150,0.4);
    }
    .model-card h3 { margin: 0 0 8px 0; font-size: 1.1rem; }
    .model-card p { margin: 0; font-size: 0.8rem; opacity: 0.85; }

    .legend-box {
        background: #f0f4ff;
        border: 1px solid #c5d5f0;
        border-radius: 7px;
        padding: 12px 16px;
        font-size: 12px;
        margin: 8px 0;
        line-height: 1.7;
    }
    .legend-box b { color: #2F5496; }

    .badge-top { background:#C6EFCE; color:#006100; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:11px; }
    .badge-mid { background:#FFEB9C; color:#9C5700; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:11px; }
    .badge-all { background:#FFC7CE; color:#9C0006; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:11px; }

    hr { margin: 10px 0; border-color: #E0E0E0; }

    .stButton > button {
        background: #2F5496 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton > button:hover { background: #1e3a5f !important; }

    /* Radio buttons as cards */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ORTAK FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(uploaded_file):
    """Excel / CSV yükle"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None


def safe_minmax(s: pd.Series) -> pd.Series:
    """Min-Max normalizasyon (0-1 arası)"""
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series([np.nan] * len(s), index=s.index)
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


# ═══════════════════════════════════════════════════════════════════════════════
# THORIUS ALGORITHM FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════

def thorius_kmeans_global(df, attribute_cols, n_clusters, desc=True):
    """Global K-Means — kapasite gruplama"""
    X = df[attribute_cols].copy()
    for col in X.columns:
        col_mean = X[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        X[col] = X[col].fillna(col_mean)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) < n_clusters:
        return np.ones(len(X), dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    means = {c: X.iloc[clusters == c].mean().mean() for c in range(n_clusters)}
    sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def thorius_calculate_weighted_score(df, metric_cols, weights):
    """Weighted Score hesapla"""
    scores = pd.DataFrame(index=df.index)

    for col in metric_cols:
        if col not in df.columns:
            continue
        values = df[col].copy()
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            scores[col] = (values - min_val) / (max_val - min_val)
        else:
            scores[col] = 0.5

    weighted_score = pd.Series(0.0, index=df.index)
    total_weight = 0

    for col in metric_cols:
        if col in scores.columns and col in weights:
            weighted_score += scores[col] * weights[col]
            total_weight += weights[col]

    if total_weight > 0:
        weighted_score = weighted_score / total_weight

    return weighted_score


def thorius_quantile_category_based(df, magaza_col, kategori_col, metric_cols, weights, n_clusters=3, desc=True):
    """Kategori bazlı K-Means performans gruplaması"""
    store_cat_df = df.groupby([magaza_col, kategori_col], as_index=False).agg(
        **{col: (col, 'sum') for col in metric_cols},
        _point_count=(metric_cols[0], 'count')
    )

    store_cat_df['_weighted_score_cat'] = thorius_calculate_weighted_score(store_cat_df, metric_cols, weights)
    store_cat_df['Urun_Grubu_Kat'] = 1

    for kategori in store_cat_df[kategori_col].unique():
        mask = store_cat_df[kategori_col] == kategori
        subset = store_cat_df.loc[mask, '_weighted_score_cat'].copy()
        subset = subset.replace([np.inf, -np.inf], np.nan)
        fill_val = subset.mean() if not pd.isna(subset.mean()) else 0
        subset = subset.fillna(fill_val)

        n_samples = len(subset)
        if n_samples < n_clusters:
            if n_samples == 0:
                continue
            ranks = subset.rank(method='first', ascending=not desc)
            store_cat_df.loc[mask, 'Urun_Grubu_Kat'] = ((ranks - 1) * n_clusters / n_samples).astype(int).clip(0, n_clusters-1) + 1
            continue

        X = subset.values.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_unique = len(np.unique(X))
        actual_clusters = min(n_clusters, n_unique, n_samples)

        if actual_clusters < 2:
            store_cat_df.loc[mask, 'Urun_Grubu_Kat'] = 1
            continue

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        means = {c: X[clusters == c].mean() for c in range(actual_clusters)}
        sorted_c = sorted(means.keys(), key=lambda x: means[x], reverse=desc)
        mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

        store_cat_df.loc[mask, 'Urun_Grubu_Kat'] = [mapping[c] for c in clusters]

    return store_cat_df[[magaza_col, kategori_col, 'Urun_Grubu_Kat', '_weighted_score_cat']]


# ═══════════════════════════════════════════════════════════════════════════════
# COPILOT V3 FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════

def copilot_kmeans_per_group(data, weighted, price_sc, cap_sc, keys, perf_w=0.7,
                              random_state=42, n_init=20):
    """Her grup için ayrı K-Means"""
    perf = pd.Series(index=data.index, dtype=object)
    cap = pd.Series(index=data.index, dtype=object)
    order_comp = pd.Series(index=data.index, dtype=float)

    for _, grp in data.groupby(keys):
        idx = grp.index

        # Performans K-Means
        X = np.c_[weighted.loc[idx].values, price_sc.loc[idx].values]
        mask = np.isfinite(X).all(axis=1)

        if mask.sum() >= 3:
            km = KMeans(n_clusters=3, random_state=random_state, n_init=n_init)
            lab = km.fit_predict(X[mask])

            comp = perf_w * weighted.loc[idx[mask]] + (1 - perf_w) * price_sc.loc[idx[mask]]
            centers = [(k, float(np.nanmean(comp.values[lab == k]))) for k in np.unique(lab)]
            order = [k for k, _ in sorted(centers, key=lambda t: t[1])]
            m = {order[0]: "Yavaş", order[1]: "Orta", order[2]: "Hızlı"}

            out = np.array(["Orta"] * len(idx), dtype=object)
            out[mask] = np.vectorize(lambda a: m.get(a, "Orta"))(lab)

            perf.loc[idx] = out
            order_comp.loc[idx[mask]] = comp.values
        else:
            comp_all = perf_w * weighted.loc[idx] + (1 - perf_w) * price_sc.loc[idx]
            order_comp.loc[idx] = comp_all.values
            try:
                q = pd.qcut(comp_all.rank(method="first"), 3, labels=["Yavaş", "Orta", "Hızlı"])
                perf.loc[idx] = q.astype(str).values
            except:
                perf.loc[idx] = "Orta"

        # Kapasite K-Means
        y = cap_sc.loc[idx].values.reshape(-1, 1)
        masky = np.isfinite(y).ravel()

        if masky.sum() >= 3:
            km2 = KMeans(n_clusters=3, random_state=random_state, n_init=n_init)
            lab2 = km2.fit_predict(y[masky])

            centers2 = [(k, float(np.nanmean(y[masky].ravel()[lab2 == k]))) for k in np.unique(lab2)]
            order2 = [k for k, _ in sorted(centers2, key=lambda t: t[1])]
            m2 = {order2[0]: "Küçük", order2[1]: "Orta", order2[2]: "Büyük"}

            out2 = np.array(["Orta"] * len(idx), dtype=object)
            out2[masky] = np.vectorize(lambda a: m2.get(a, "Orta"))(lab2)

            cap.loc[idx] = out2
        else:
            try:
                q2 = pd.qcut(cap_sc.loc[idx].rank(method="first"), 3, labels=["Küçük", "Orta", "Büyük"])
                cap.loc[idx] = q2.astype(str).values
            except:
                cap.loc[idx] = "Orta"

    return perf, cap, order_comp


def copilot_assign_cluster_code(cap, perf):
    """Kapasite + Performans → Cluster Code (Yeni İsimlendirme)"""
    concat_key = (cap.astype(str) + perf.astype(str)).replace("nannan", "")

    # Yeni mapping: TOP=Hızlı, MID=Orta performans, ALL=Yavaş
    pair_to_code = {
        ("Büyük", "Hızlı"): "TOP1", ("Orta", "Hızlı"): "TOP2",  ("Büyük", "Orta"): "TOP3",
        ("Küçük", "Hızlı"): "MID4", ("Orta", "Orta"): "MID5",   ("Küçük", "Orta"): "MID6",
        ("Büyük", "Yavaş"): "ALL7", ("Orta", "Yavaş"): "ALL8",  ("Küçük", "Yavaş"): "ALL9",
    }

    code = pd.Series([pair_to_code.get((c, p), np.nan) for c, p in zip(cap, perf)], index=cap.index)
    detail = concat_key.map(lambda k: COPILOT_KEY_TO_DETAIL.get(k, np.nan))

    return code, detail, concat_key


def copilot_min10_balancing(df, keys, score_col="_order_comp", cap_sc_col="Capacity_sc", min_per_rank=10):
    """Min-10 Balancing"""
    base_rank = df["CO_PilotCluster"].map(lambda c: COPILOT_CODE_TO_RANK.get(c, np.nan))

    score_med = df[score_col].median() if df[score_col].notna().any() else 0
    cap_med = df[cap_sc_col].median() if df[cap_sc_col].notna().any() else 0
    combined_score = (df[score_col].fillna(score_med) + df[cap_sc_col].fillna(cap_med)) / 2

    adj_rank = base_rank.copy()
    move_log = []

    for gk, grp in df.groupby(keys):
        idx = grp.index
        n = len(idx)

        if n < min_per_rank * 9:
            continue

        ranks = adj_rank.loc[idx].copy()
        scores = combined_score.loc[idx].copy()

        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            counts = ranks.value_counts()

            deficits = {r: min_per_rank - int(counts.get(r, 0))
                        for r in range(1, 10)
                        if int(counts.get(r, 0)) < min_per_rank}

            if not deficits:
                break

            donors = {r: int(counts.get(r, 0)) - min_per_rank
                      for r in range(1, 10)
                      if int(counts.get(r, 0)) > min_per_rank}

            if not donors:
                break

            r_target = sorted(deficits.items(), key=lambda t: t[1], reverse=True)[0][0]
            need = deficits[r_target]

            donor_candidates = sorted(donors.keys(), key=lambda r: (abs(r - r_target), r))
            if not donor_candidates:
                break

            r_donor = donor_candidates[0]
            take = min(need, donors[r_donor])

            donor_idx = ranks[ranks == r_donor].index

            if r_donor < r_target:
                pick = scores.loc[donor_idx].sort_values(ascending=True).head(take).index
            else:
                pick = scores.loc[donor_idx].sort_values(ascending=False).head(take).index

            ranks.loc[pick] = r_target

            move_log.append({
                "Group": str(gk),
                "from_rank": r_donor,
                "to_rank": r_target,
                "moved_count": len(pick)
            })

        adj_rank.loc[idx] = ranks

    adj_code = adj_rank.map(lambda r: COPILOT_RANK_TO_CODE.get(int(r), np.nan) if pd.notna(r) else np.nan)
    adj_pair = adj_rank.map(lambda r: COPILOT_RANK_TO_PAIR.get(int(r), (np.nan, np.nan)) if pd.notna(r) else (np.nan, np.nan))
    adj_cap = adj_pair.map(lambda t: t[0])
    adj_perf = adj_pair.map(lambda t: t[1])

    adj_key = (adj_cap.astype(str) + adj_perf.astype(str)).replace("nannan", "")
    adj_detail = adj_key.map(lambda k: COPILOT_KEY_TO_DETAIL.get(k, np.nan))

    return adj_perf, adj_cap, adj_code, adj_detail, pd.DataFrame(move_log)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Session state init
    for key in ['model', 'data_df', 'kapasite_df', 'urun_df', 'results', 'config', 'move_log']:
        if key not in st.session_state:
            st.session_state[key] = None

    # Splash
    if 'splash_shown' not in st.session_state:
        st.session_state.splash_shown = True
        st.markdown("""
        <div class="splash-overlay">
            <div class="splash-title">📊 Cluster <span>Analizi</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL SEÇİMİ (Ana Sayfa + Sidebar)
    # ═══════════════════════════════════════════════════════════════════════════

    # Ana başlık
    st.markdown("""
    <div class="app-header">
        <h1>📊 Cluster <span>Analizi</span></h1>
        <div class="sub">Thorius Algorithm & CoPilot V3 | K-Means Clustering</div>
    </div>
    """, unsafe_allow_html=True)

    # Model seçimi - ANA SAYFADA
    st.markdown("### 🎯 Algoritma Seçin")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        thorius_selected = st.button(
            "📦 Thorius Algorithm",
            use_container_width=True,
            type="primary" if st.session_state.get('model') != "CoPilot V3" else "secondary",
            key='btn_thorius'
        )
        st.caption("Kapasite + Kategori bazlı | 1-1 → 3-3")

    with col_m2:
        copilot_selected = st.button(
            "🎯 CoPilot V3",
            use_container_width=True,
            type="primary" if st.session_state.get('model') == "CoPilot V3" else "secondary",
            key='btn_copilot'
        )
        st.caption("LivingArea bazlı | TOP/MID/ALL + Min-10")

    # Buton tıklamalarını işle
    if thorius_selected:
        st.session_state.model = "Thorius Algorithm"
        st.rerun()
    if copilot_selected:
        st.session_state.model = "CoPilot V3"
        st.rerun()

    # Varsayılan model
    if 'model' not in st.session_state or st.session_state.model is None:
        st.session_state.model = "Thorius Algorithm"

    model = st.session_state.model

    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar'da da göster
    with st.sidebar:
        st.markdown("## 🎯 Aktif Model")
        st.info(f"**{model}**")

        st.markdown("<hr>", unsafe_allow_html=True)

        if model == "Thorius Algorithm":
            st.markdown("""
            **📦 Thorius Algorithm**

            - Global K-Means Kapasite
            - Kategori bazlı Ürün gruplaması
            - 9 Kombine Grup: 1-1 → 3-3
            - Weighted Score (özelleştirilebilir)
            """)
        else:
            st.markdown("""
            **🎯 CoPilot V3**

            - LivingArea + MainGroup + SubGroup
            - Weighted: 0.2/0.3/0.5 (Adet/Ciro/Kar)
            - Performans: Yavaş/Orta/Hızlı
            - Min-10 Balancing
            - 9 Cluster: TOP1-3, MID1-3, ALL1-3
            """)

    # ═══════════════════════════════════════════════════════════════════════════
    # THORIUS ALGORITHM
    # ═══════════════════════════════════════════════════════════════════════════
    if model == "Thorius Algorithm":
        st.markdown("### 📦 Thorius Algorithm")

        col_left, col_right = st.columns([1, 2.5])

        with col_left:
            # Tek dosya yükleme
            st.markdown('<div class="section-header">📁 Veri Yükleme</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Excel dosyası", type=['xlsx', 'xls', 'csv'],
                                        key='thorius_upload', label_visibility="collapsed")

            if uploaded:
                df = load_data(uploaded)
                if df is not None:
                    st.session_state.data_df = df
                    st.success(f"✅ {len(df):,} satır yüklendi")

            if st.session_state.data_df is not None:
                df = st.session_state.data_df
                all_cols = df.columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                st.markdown("<hr>", unsafe_allow_html=True)

                # Mağaza & Kategori
                st.markdown('<div class="section-header-blue">🏪 Temel Kolonlar</div>', unsafe_allow_html=True)

                def find_idx(name, cols):
                    matches = [i for i, c in enumerate(cols) if name.lower() in c.lower()]
                    return matches[0] if matches else 0

                magaza_col = st.selectbox("🏪 Mağaza Kolonu", all_cols,
                                          index=find_idx("store", all_cols), key='th_magaza')
                kategori_col = st.selectbox("🏷️ Kategori Kolonu",
                                            [c for c in all_cols if c != magaza_col],
                                            key='th_kategori')

                st.markdown("<hr>", unsafe_allow_html=True)

                # Kapasite Ayarları
                st.markdown('<div class="section-header">📦 Kapasite Gruplama</div>', unsafe_allow_html=True)

                kap_attrs = st.multiselect("Kapasite Kolonları (m², dm³, vb.)",
                                           [c for c in numeric_cols if c != magaza_col],
                                           default=[c for c in numeric_cols if 'capacity' in c.lower()][:2],
                                           key='th_kap_attrs')
                kap_grup = st.number_input("Kapasite Grup Sayısı", 2, 10, 3, key='th_kap_grup')

                st.markdown("<hr>", unsafe_allow_html=True)

                # Ürün/Performans Ayarları
                st.markdown('<div class="section-header">📊 Ürün/Performans Gruplama</div>', unsafe_allow_html=True)

                available_metrics = [c for c in numeric_cols if c not in [magaza_col] + kap_attrs]
                urun_metrics = st.multiselect("Performans Metrikleri (Satış, Ciro, Kar vb.)",
                                              available_metrics,
                                              default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
                                              key='th_urun_metrics')
                urun_grup = st.number_input("Ürün Grup Sayısı", 2, 10, 3, key='th_urun_grup')

                st.markdown("<hr>", unsafe_allow_html=True)

                # Ağırlıklar
                st.markdown('<div class="section-header-green">⚖️ Metrik Ağırlıkları</div>', unsafe_allow_html=True)
                weights = {}
                default_weights = [40, 35, 25, 20, 15]  # İlk 5 metrik için varsayılan
                for i, col in enumerate(urun_metrics[:5]):
                    w = st.slider(f"{col[:25]}", 0, 100, default_weights[i] if i < len(default_weights) else 20, key=f'th_w_{i}')
                    weights[col] = w

                total_w = sum(weights.values())
                if total_w > 0:
                    st.caption(f"Toplam: {total_w} → Normalize edilecek")

                st.markdown("<hr>", unsafe_allow_html=True)

                # Sıralama
                desc_order = st.checkbox("DESC (Büyük/Yüksek → 1)", value=True, key='th_desc')

                st.markdown("<hr>", unsafe_allow_html=True)

                if st.button("🚀 Hesapla", use_container_width=True, type="primary", key='th_run'):
                    if kap_attrs and urun_metrics:
                        with st.spinner("Hesaplanıyor..."):
                            try:
                                data = df.copy()

                                # Numerik dönüşüm
                                for c in kap_attrs + urun_metrics:
                                    data[c] = pd.to_numeric(data[c], errors='coerce')

                                # 1. Kapasite Gruplama (mağaza bazında unique)
                                kap_agg = {c: 'max' for c in kap_attrs}
                                kap_df = data.groupby(magaza_col, as_index=False).agg(kap_agg)
                                kap_df['Kapasite_Grubu'] = thorius_kmeans_global(kap_df, kap_attrs, kap_grup, desc=desc_order)

                                st.info(f"📦 {len(kap_df)} unique mağaza → {kap_grup} kapasite grubu")

                                # 2. Ürün/Performans Gruplama (kategori bazında)
                                store_cat_groups = thorius_quantile_category_based(
                                    data, magaza_col, kategori_col, urun_metrics, weights, urun_grup, desc_order
                                )

                                # 3. Birleştir
                                result = data.merge(kap_df[[magaza_col, 'Kapasite_Grubu']], on=magaza_col, how='left')
                                result = result.merge(store_cat_groups, on=[magaza_col, kategori_col], how='left')

                                result['Kapasite_Grubu'] = result['Kapasite_Grubu'].fillna('?')
                                result['Urun_Grubu'] = result['Urun_Grubu_Kat'].fillna(1).astype(int)
                                result['Kombine_Grup'] = result['Kapasite_Grubu'].astype(str) + '-' + result['Urun_Grubu'].astype(str)

                                # Yeni isimlendirme kolonları ekle
                                result['Cluster_Kod'] = result['Kombine_Grup'].map(THORIUS_KOMBINE_TO_SHORT).fillna(result['Kombine_Grup'])
                                result['Cluster_Detay'] = result['Kombine_Grup'].map(THORIUS_KOMBINE_TO_CODE).fillna(result['Kombine_Grup'])

                                st.session_state.results = result
                                st.session_state.config = {
                                    'magaza_col': magaza_col,
                                    'kategori_col': kategori_col,
                                    'metrics': urun_metrics,
                                    'kap_attrs': kap_attrs,
                                }
                                st.success("✅ Tamamlandı!")
                            except Exception as e:
                                st.error(f"❌ Hata: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ Kapasite ve Metrik kolonlarını seçin!")

        with col_right:
            st.markdown('<div class="section-header-blue">📊 Sonuçlar</div>', unsafe_allow_html=True)

            if st.session_state.results is not None:
                results = st.session_state.results
                cfg = st.session_state.config

                # Legend
                st.markdown("""
                <div class="legend-box">
                    <span class="badge-top">TOP1-3</span> Hızlı Performans |
                    <span class="badge-mid">MID4-6</span> Orta Performans |
                    <span class="badge-all">ALL7-9</span> Yavaş Performans<br>
                    <b>Detay:</b> TOP1=Büyük-Hızlı, TOP2=Orta-Hızlı, TOP3=Büyük-Orta |
                    MID4=Küçük-Hızlı, MID5=Orta-Orta, MID6=Küçük-Orta |
                    ALL7=Büyük-Yavaş, ALL8=Orta-Yavaş, ALL9=Küçük-Yavaş
                </div>
                """, unsafe_allow_html=True)

                # KPI
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Satır", f"{len(results):,}")

                magaza_col = cfg.get('magaza_col', None)
                kategori_col = cfg.get('kategori_col', None)

                if magaza_col and magaza_col in results.columns:
                    c2.metric("Mağaza", f"{results[magaza_col].nunique():,}")
                else:
                    c2.metric("Mağaza", "-")

                if kategori_col and kategori_col in results.columns:
                    c3.metric("Kategori", f"{results[kategori_col].nunique():,}")
                else:
                    c3.metric("Kategori", "-")

                if 'Cluster_Kod' in results.columns:
                    c4.metric("Cluster", f"{results['Cluster_Kod'].nunique()}")
                else:
                    c4.metric("Cluster", "-")

                st.markdown("<hr>", unsafe_allow_html=True)

                # Kategori Filtresi
                if kategori_col and kategori_col in results.columns:
                    kategoriler = ['🔄 Tümü'] + sorted(results[kategori_col].dropna().unique().tolist())
                    selected_kat = st.selectbox("🏷️ Kategori Filtre", kategoriler, key='th_filter')

                    if selected_kat != '🔄 Tümü':
                        filtered = results[results[kategori_col] == selected_kat]
                    else:
                        filtered = results
                else:
                    filtered = results

                st.markdown("<hr>", unsafe_allow_html=True)

                # Grup Dağılımı
                st.markdown("**📊 Cluster Dağılımı**")
                if 'Cluster_Kod' in filtered.columns:
                    counts = filtered['Cluster_Kod'].value_counts().reset_index()
                    counts.columns = ['Cluster', 'Count']

                    # Sıralama: TOP1, TOP2, TOP3, MID4, MID5, MID6, ALL7, ALL8, ALL9
                    order = ['TOP1', 'TOP2', 'TOP3', 'MID4', 'MID5', 'MID6', 'ALL7', 'ALL8', 'ALL9']
                    counts['sort'] = counts['Cluster'].apply(lambda x: order.index(x) if x in order else 99)
                    counts = counts.sort_values('sort').drop('sort', axis=1)

                    # Renk haritası
                    color_map = {
                        'TOP1': '#006100', 'TOP2': '#38761d', 'TOP3': '#6aa84f',
                        'MID4': '#9C5700', 'MID5': '#b8860b', 'MID6': '#daa520',
                        'ALL7': '#9C0006', 'ALL8': '#cc0000', 'ALL9': '#ea9999',
                    }

                    fig = px.bar(counts, x='Cluster', y='Count', color='Cluster',
                                 color_discrete_map=color_map, height=300)
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                # Özet Tablo
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("**📋 Cluster Özet Tablosu**")

                summary_rows = []
                for grp in order:
                    if 'Cluster_Kod' not in filtered.columns:
                        break
                    grp_data = filtered[filtered['Cluster_Kod'] == grp]
                    if len(grp_data) == 0:
                        continue

                    # Detay bilgisi al
                    detay = grp_data['Cluster_Detay'].iloc[0] if 'Cluster_Detay' in grp_data.columns and len(grp_data) > 0 else grp

                    row = {
                        'Cluster': detay,
                        'Satır': f"{len(grp_data):,}".replace(",", "."),
                    }

                    if magaza_col and magaza_col in grp_data.columns:
                        row['Mağaza'] = f"{grp_data[magaza_col].nunique():,}".replace(",", ".")

                    # Metrik ortalamaları
                    metrics = cfg.get('metrics', [])
                    for metric in metrics[:3]:
                        if metric in grp_data.columns:
                            avg = grp_data[metric].mean()
                            row[metric[:15]] = f"{int(avg):,}".replace(",", ".")

                    # Weighted score
                    if '_weighted_score_cat' in grp_data.columns:
                        ws = grp_data['_weighted_score_cat'].mean()
                        row['W.Score'] = f"{ws:.3f}"

                    summary_rows.append(row)

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

                # 3D Scatter (eğer kapasite kolonu varsa)
                kap_attrs = cfg.get('kap_attrs', [])
                if kap_attrs and len(kap_attrs) > 0 and 'Cluster_Kod' in filtered.columns:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("**🎯 3D Scatter**")

                    kap_col = kap_attrs[0]
                    metric_col = metrics[0] if metrics else None

                    if metric_col and kap_col in filtered.columns and metric_col in filtered.columns:
                        sample_df = filtered.sample(min(3000, len(filtered)))

                        fig3d = px.scatter_3d(
                            sample_df,
                            x=kap_col,
                            y=metric_col,
                            z='_weighted_score_cat' if '_weighted_score_cat' in sample_df.columns else metric_col,
                            color='Cluster_Kod',
                            color_discrete_map=color_map,
                            opacity=0.7,
                            height=450
                        )
                        fig3d.update_traces(marker=dict(size=4))
                        fig3d.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig3d, use_container_width=True)

                # İndir
                st.markdown("<hr>", unsafe_allow_html=True)
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    buffer = io.BytesIO()
                    results.to_excel(buffer, index=False, engine='openpyxl')
                    st.download_button("📥 Excel (Tümü)", buffer.getvalue(),
                                       f"thorius_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                       use_container_width=True)

                with col_d2:
                    csv = filtered.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 CSV (Filtreli)", csv,
                                       f"thorius_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                       use_container_width=True)

                # Detaylı Tablo
                with st.expander("📋 Tüm Veriyi Göster", expanded=False):
                    show_cols = []
                    if magaza_col:
                        show_cols.append(magaza_col)
                    if kategori_col:
                        show_cols.append(kategori_col)
                    show_cols += ['Cluster_Kod', 'Cluster_Detay', 'Kapasite_Grubu', 'Urun_Grubu']
                    show_cols += cfg.get('metrics', [])[:3]
                    show_cols = [c for c in show_cols if c in results.columns]
                    if show_cols:
                        st.dataframe(results[show_cols], height=400, use_container_width=True)
                    else:
                        st.dataframe(results, height=400, use_container_width=True)

            else:
                st.info("👈 Sol panelden veri yükleyip hesaplayın")

                # Demo
                st.markdown("**📊 Demo — Örnek Cluster Dağılımı**")
                demo = pd.DataFrame({
                    'Cluster': ['TOP1', 'TOP2', 'TOP3', 'MID4', 'MID5', 'MID6', 'ALL7', 'ALL8', 'ALL9'],
                    'Count': [120, 150, 100, 180, 220, 160, 250, 300, 220]
                })
                fig = px.bar(demo, x='Cluster', y='Count',
                             color='Cluster',
                             color_discrete_sequence=['#006100', '#38761d', '#6aa84f',
                                                      '#9C5700', '#b8860b', '#daa520',
                                                      '#9C0006', '#cc0000', '#ea9999'],
                             height=300)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # COPILOT V3
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 🎯 CoPilot V3")

        col_left, col_right = st.columns([1, 2.5])

        with col_left:
            st.markdown('<div class="section-header">📁 Veri Yükleme</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Excel", type=['xlsx'], key='copilot_upload', label_visibility="collapsed")

            if uploaded:
                try:
                    xls = pd.ExcelFile(uploaded, engine='openpyxl')
                    sheet = st.selectbox("Sayfa", xls.sheet_names, key='copilot_sheet')
                    st.session_state.data_df = xls.parse(sheet)
                    st.success(f"✅ {len(st.session_state.data_df)} satır")
                except Exception as e:
                    st.error(f"❌ {e}")

            if st.session_state.data_df is not None:
                df = st.session_state.data_df
                all_cols = df.columns.tolist()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-header-blue">🏷️ Grup Kolonları</div>', unsafe_allow_html=True)

                def find_idx(name, cols):
                    matches = [i for i, c in enumerate(cols) if name.lower() in c.lower()]
                    return matches[0] if matches else 0

                col_living = st.selectbox("LivingArea", all_cols, index=find_idx("living", all_cols), key='cp_living')
                col_main = st.selectbox("MainGroupDesc", all_cols, index=find_idx("maingroup", all_cols), key='cp_main')
                col_sub = st.selectbox("SubGroupDesc", all_cols, index=find_idx("subgroup", all_cols), key='cp_sub')

                st.markdown('<div class="section-header-blue">📊 Metrikler</div>', unsafe_allow_html=True)
                col_e = st.selectbox("Satış Adet", numeric_cols, index=find_idx("unit", numeric_cols), key='cp_e')
                col_f = st.selectbox("Ciro", numeric_cols, index=find_idx("value", numeric_cols), key='cp_f')
                col_g = st.selectbox("Brüt Kar", numeric_cols, index=find_idx("profit", numeric_cols), key='cp_g')
                col_i = st.selectbox("Kapasite", numeric_cols, index=find_idx("capacity", numeric_cols), key='cp_i')
                col_j = st.selectbox("Fiyat", numeric_cols, index=find_idx("price", numeric_cols), key='cp_j')

                st.markdown('<div class="section-header-green">⚖️ Ağırlıklar</div>', unsafe_allow_html=True)
                w_unit = st.slider("Satış Adet", 0, 100, 20, key='cp_w1')
                w_value = st.slider("Ciro", 0, 100, 30, key='cp_w2')
                w_profit = st.slider("Brüt Kar", 0, 100, 50, key='cp_w3')
                perf_w = st.slider("Perf Composite (Weighted vs Price)", 0.0, 1.0, 0.7, key='cp_pw')

                st.markdown('<div class="section-header-orange">📏 Min-10 Balancing</div>', unsafe_allow_html=True)
                enable_min10 = st.checkbox("Min-10 Uygula", True, key='cp_min10')
                min_rank = st.number_input("Min mağaza/rank", 1, 50, 10, key='cp_minrank')

                st.markdown("<hr>", unsafe_allow_html=True)

                if st.button("🚀 Hesapla", use_container_width=True, type="primary", key='cp_run'):
                    with st.spinner("Hesaplanıyor..."):
                        try:
                            data = df.copy()
                            keys = [col_living, col_main, col_sub]

                            for c in [col_e, col_f, col_g, col_i, col_j]:
                                data[c] = pd.to_numeric(data[c], errors='coerce')

                            E_sc = data.groupby(keys)[col_e].transform(safe_minmax)
                            F_sc = data.groupby(keys)[col_f].transform(safe_minmax)
                            G_sc = data.groupby(keys)[col_g].transform(safe_minmax)
                            J_sc = data.groupby(keys)[col_j].transform(safe_minmax)
                            I_sc = data.groupby(keys)[col_i].transform(safe_minmax)

                            total = w_unit + w_value + w_profit
                            weighted = (w_unit/total)*E_sc + (w_value/total)*F_sc + (w_profit/total)*G_sc if total > 0 else E_sc

                            perf, cap, order_comp = copilot_kmeans_per_group(data, weighted, J_sc, I_sc, keys, perf_w)
                            code, detail, _ = copilot_assign_cluster_code(cap, perf)

                            data["Weighted_Score"] = weighted
                            data["Capacity_sc"] = I_sc
                            data["_order_comp"] = order_comp
                            data["PerformansGrade"] = perf
                            data["KapasiteCluster"] = cap
                            data["CO_PilotCluster"] = code
                            data["CO_PilotCluster_Detay"] = detail

                            move_log = None
                            if enable_min10:
                                adj_perf, adj_cap, adj_code, adj_detail, move_log = copilot_min10_balancing(
                                    data, keys, "_order_comp", "Capacity_sc", min_rank
                                )
                                data["PerformansGrade_min10"] = adj_perf
                                data["KapasiteCluster_min10"] = adj_cap
                                data["CO_PilotCluster_min10"] = adj_code

                            st.session_state.results = data
                            st.session_state.move_log = move_log
                            st.session_state.config = {
                                'keys': keys,
                                'enable_min10': enable_min10,
                                'col_e': col_e,  # Satış Adet
                                'col_f': col_f,  # Ciro
                                'col_g': col_g,  # Brüt Kar
                                'col_i': col_i,  # Kapasite
                                'col_j': col_j,  # Fiyat
                            }
                            st.success("✅ Tamamlandı!")
                        except Exception as e:
                            st.error(f"❌ {e}")
                            import traceback
                            st.code(traceback.format_exc())

        with col_right:
            st.markdown('<div class="section-header-blue">📊 Sonuçlar</div>', unsafe_allow_html=True)

            if st.session_state.results is not None and st.session_state.config is not None:
                results = st.session_state.results
                cfg = st.session_state.config

                enable_min10 = cfg.get('enable_min10', False)
                cluster_col = "CO_PilotCluster_min10" if enable_min10 and "CO_PilotCluster_min10" in results.columns else "CO_PilotCluster"

                # Fallback if cluster_col doesn't exist
                if cluster_col not in results.columns:
                    cluster_col = [c for c in results.columns if 'Cluster' in c or 'cluster' in c]
                    cluster_col = cluster_col[0] if cluster_col else None

                # Legend
                st.markdown("""
                <div class="legend-box">
                    <span class="badge-top">TOP1-3</span> Hızlı Performans |
                    <span class="badge-mid">MID4-6</span> Orta Performans |
                    <span class="badge-all">ALL7-9</span> Yavaş Performans<br>
                    <b>Detay:</b> TOP1=Büyük-Hızlı, TOP2=Orta-Hızlı, TOP3=Büyük-Orta |
                    MID4=Küçük-Hızlı, MID5=Orta-Orta, MID6=Küçük-Orta |
                    ALL7=Büyük-Yavaş, ALL8=Orta-Yavaş, ALL9=Küçük-Yavaş
                </div>
                """, unsafe_allow_html=True)

                # KPI
                c1, c2, c3 = st.columns(3)
                c1.metric("Satır", f"{len(results):,}")

                keys = cfg.get('keys', [])
                if keys and all(k in results.columns for k in keys):
                    c2.metric("Grup", f"{results[keys].drop_duplicates().shape[0]:,}")
                else:
                    c2.metric("Grup", "-")

                if cluster_col and cluster_col in results.columns:
                    c3.metric("Cluster", f"{results[cluster_col].nunique()}")
                else:
                    c3.metric("Cluster", "-")

                # Filtreler
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-header-orange">🔍 Filtreler</div>', unsafe_allow_html=True)
                keys = cfg.get('keys', [])

                filtered = results.copy()

                # Filtre seçenekleri - 3 kolon halinde
                if keys and len(keys) >= 3:
                    col_f1, col_f2, col_f3 = st.columns(3)

                    with col_f1:
                        # LivingArea filtresi
                        living_col = keys[0]
                        living_opts = ['🔄 Tümü'] + sorted(results[living_col].dropna().unique().tolist())
                        selected_living = st.selectbox(f"🏠 {living_col}", living_opts, key='cp_filter_living')

                    with col_f2:
                        # MainGroup filtresi
                        main_col = keys[1]
                        main_opts = ['🔄 Tümü'] + sorted(results[main_col].dropna().unique().tolist())
                        selected_main = st.selectbox(f"📦 {main_col}", main_opts, key='cp_filter_main')

                    with col_f3:
                        # SubGroup filtresi
                        sub_col = keys[2]
                        sub_opts = ['🔄 Tümü'] + sorted(results[sub_col].dropna().unique().tolist())
                        selected_sub = st.selectbox(f"🏷️ {sub_col}", sub_opts, key='cp_filter_sub')

                    # Filtreleri uygula
                    if selected_living != '🔄 Tümü':
                        filtered = filtered[filtered[living_col] == selected_living]
                    if selected_main != '🔄 Tümü':
                        filtered = filtered[filtered[main_col] == selected_main]
                    if selected_sub != '🔄 Tümü':
                        filtered = filtered[filtered[sub_col] == selected_sub]

                    # Filtre sonucu bilgisi
                    if len(filtered) < len(results):
                        st.caption(f"📊 Filtrelenmiş: {len(filtered):,} / {len(results):,} satır ({100*len(filtered)/len(results):.1f}%)")

                # Dağılım
                if cluster_col and cluster_col in filtered.columns:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    counts = filtered[cluster_col].value_counts().reset_index()
                    counts.columns = ['Cluster', 'Count']

                    order = ['TOP1', 'TOP2', 'TOP3', 'MID4', 'MID5', 'MID6', 'ALL7', 'ALL8', 'ALL9']
                    counts['sort'] = counts['Cluster'].apply(lambda x: order.index(x) if x in order else 99)
                    counts = counts.sort_values('sort').drop('sort', axis=1)

                    color_map = {
                        'TOP1': '#006100', 'TOP2': '#38761d', 'TOP3': '#6aa84f',
                        'MID4': '#9C5700', 'MID5': '#b8860b', 'MID6': '#daa520',
                        'ALL7': '#9C0006', 'ALL8': '#cc0000', 'ALL9': '#ea9999',
                    }

                    fig = px.bar(counts, x='Cluster', y='Count', color='Cluster', color_discrete_map=color_map, height=300)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # ═══════════════════════════════════════════════════════════════════════════
                # DETAYLI İSTATİSTİK TABLOSU
                # ═══════════════════════════════════════════════════════════════════════════
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-header-green">📊 Detaylı Cluster İstatistikleri</div>', unsafe_allow_html=True)

                # Metrik kolonlarını al
                col_e = cfg.get('col_e')  # Satış Adet
                col_f = cfg.get('col_f')  # Ciro
                col_g = cfg.get('col_g')  # Brüt Kar
                col_i = cfg.get('col_i')  # Kapasite
                col_j = cfg.get('col_j')  # Fiyat

                if cluster_col and cluster_col in filtered.columns:
                    # Mağaza kolonu bul (ilk key genellikle store/mağaza)
                    store_col = keys[0] if keys else None

                    # Cluster bazında istatistikler
                    stats_rows = []
                    cluster_order = ['TOP1', 'TOP2', 'TOP3', 'MID4', 'MID5', 'MID6', 'ALL7', 'ALL8', 'ALL9']

                    # Global ortalama ve varyans (gruplar arası hesaplama için)
                    global_mean_score = filtered['Weighted_Score'].mean() if 'Weighted_Score' in filtered.columns else 0

                    # Her cluster için istatistik hesapla
                    cluster_means = {}
                    cluster_vars = {}

                    for clust in cluster_order:
                        clust_data = filtered[filtered[cluster_col] == clust]
                        if len(clust_data) == 0:
                            continue

                        # Temel sayılar
                        count = len(clust_data)
                        n_stores = clust_data[store_col].nunique() if store_col and store_col in clust_data.columns else 0

                        # Ortalamalar
                        avg_capacity = clust_data[col_i].mean() if col_i and col_i in clust_data.columns else 0
                        avg_unit = clust_data[col_e].mean() if col_e and col_e in clust_data.columns else 0
                        avg_revenue = clust_data[col_f].mean() if col_f and col_f in clust_data.columns else 0
                        avg_profit = clust_data[col_g].mean() if col_g and col_g in clust_data.columns else 0
                        avg_price = clust_data[col_j].mean() if col_j and col_j in clust_data.columns else 0

                        # Weighted Score ortalaması ve varyansı
                        ws_mean = clust_data['Weighted_Score'].mean() if 'Weighted_Score' in clust_data.columns else 0
                        ws_var = clust_data['Weighted_Score'].var() if 'Weighted_Score' in clust_data.columns else 0

                        cluster_means[clust] = ws_mean
                        cluster_vars[clust] = ws_var

                        # Grup içi varyans (within-cluster variance)
                        within_var = ws_var if not pd.isna(ws_var) else 0

                        stats_rows.append({
                            'Cluster': clust,
                            'Count': count,
                            'Mağaza': n_stores,
                            'Kapasite': avg_capacity,
                            'Satış Adet': avg_unit,
                            'Ciro': avg_revenue,
                            'Brüt Kar': avg_profit,
                            'Fiyat': avg_price,
                            'Grup İçi Var.': within_var,
                        })

                    if stats_rows:
                        stats_df = pd.DataFrame(stats_rows)

                        # Gruplar arası varyans hesapla (between-cluster variance)
                        if cluster_means:
                            overall_mean = np.mean(list(cluster_means.values()))
                            between_var = np.var(list(cluster_means.values()))
                        else:
                            between_var = 0

                        # Küme güvenilirliği hesapla
                        # Silhouette-benzeri metrik: between_var / (between_var + avg_within_var)
                        avg_within_var = stats_df['Grup İçi Var.'].mean() if len(stats_df) > 0 else 0

                        stats_df['Gruplar Arası Var.'] = between_var

                        # Güvenilirlik: Yüksek between_var ve düşük within_var = iyi ayrım
                        if (between_var + avg_within_var) > 0:
                            reliability = (between_var / (between_var + avg_within_var)) * 100
                        else:
                            reliability = 50.0

                        # Her cluster için bireysel güvenilirlik
                        def calc_cluster_reliability(row):
                            within = row['Grup İçi Var.'] if not pd.isna(row['Grup İçi Var.']) else 0
                            if (between_var + within) > 0:
                                return (between_var / (between_var + within)) * 100
                            return 50.0

                        stats_df['Güvenilirlik %'] = stats_df.apply(calc_cluster_reliability, axis=1)

                        # Formatla
                        display_df = stats_df.copy()
                        display_df['Count'] = display_df['Count'].apply(lambda x: f"{int(x):,}".replace(",", "."))
                        display_df['Mağaza'] = display_df['Mağaza'].apply(lambda x: f"{int(x):,}".replace(",", "."))
                        display_df['Kapasite'] = display_df['Kapasite'].apply(lambda x: f"{x:,.1f}".replace(",", "."))
                        display_df['Satış Adet'] = display_df['Satış Adet'].apply(lambda x: f"{x:,.1f}".replace(",", "."))
                        display_df['Ciro'] = display_df['Ciro'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
                        display_df['Brüt Kar'] = display_df['Brüt Kar'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
                        display_df['Fiyat'] = display_df['Fiyat'].apply(lambda x: f"{x:,.2f}".replace(",", "."))
                        display_df['Grup İçi Var.'] = display_df['Grup İçi Var.'].apply(lambda x: f"{x:.4f}")
                        display_df['Gruplar Arası Var.'] = display_df['Gruplar Arası Var.'].apply(lambda x: f"{x:.4f}")
                        display_df['Güvenilirlik %'] = display_df['Güvenilirlik %'].apply(lambda x: f"{x:.1f}%")

                        st.dataframe(display_df, hide_index=True, use_container_width=True)

                        # Özet metrikler
                        col_s1, col_s2, col_s3 = st.columns(3)
                        col_s1.metric("Ortalama Grup İçi Varyans", f"{avg_within_var:.4f}")
                        col_s2.metric("Gruplar Arası Varyans", f"{between_var:.4f}")
                        col_s3.metric("Genel Küme Güvenilirliği", f"{reliability:.1f}%")

                        # Güvenilirlik açıklaması
                        if reliability >= 70:
                            st.success(f"✅ Kümeleme kalitesi YÜKSEK - Clusterlar birbirinden iyi ayrılmış")
                        elif reliability >= 50:
                            st.warning(f"⚠️ Kümeleme kalitesi ORTA - Bazı clusterlar örtüşüyor olabilir")
                        else:
                            st.error(f"❌ Kümeleme kalitesi DÜŞÜK - Clusterlar yeterince ayrışmamış")

                # İndir
                st.markdown("<hr>", unsafe_allow_html=True)
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        results.to_excel(writer, index=False, sheet_name='CoPilot_V3')
                        if st.session_state.move_log is not None and len(st.session_state.move_log) > 0:
                            st.session_state.move_log.to_excel(writer, index=False, sheet_name='Min10_Log')

                    st.download_button("📥 Excel İndir", buffer.getvalue(),
                                       f"copilot_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                       use_container_width=True)

                with col_d2:
                    csv = filtered.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 CSV İndir", csv,
                                       f"copilot_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                       use_container_width=True)

                # Detaylı Tablo
                with st.expander("📋 Tüm Veriyi Göster", expanded=False):
                    show_cols = keys.copy() if keys else []
                    show_cols += ['PerformansGrade', 'KapasiteCluster', cluster_col]
                    if 'Weighted_Score' in filtered.columns:
                        show_cols.append('Weighted_Score')
                    show_cols = [c for c in show_cols if c in filtered.columns]
                    if show_cols:
                        st.dataframe(filtered[show_cols], height=400, use_container_width=True)
                    else:
                        st.dataframe(filtered, height=400, use_container_width=True)
            else:
                st.info("👈 Sol panelden veri yükleyip hesaplayın")


if __name__ == "__main__":
    main()
