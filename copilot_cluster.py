"""
CoPilot Cluster V3 — LivingArea + MainGroupDesc + SubGroupDesc Bazlı
Weighted Score (0.2/0.3/0.5) + Price K-Means + Min-10 Balancing

Metodoloji:
1. Group Min-Max ölçekleme (her LivingArea-MainGroup-SubGroup için ayrı)
2. Weighted Score = 0.2*SalesUnit + 0.3*SalesValue + 0.5*GrossProfit
3. Performans K-Means: [Weighted, Price] → Yavaş/Orta/Hızlı
4. Kapasite K-Means: [Capacity] → Küçük/Orta/Büyük
5. Min-10 Balancing: Her rank için minimum 10 mağaza garantisi
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import io
from datetime import datetime

st.set_page_config(page_title="CoPilot Cluster V3", page_icon="🎯", layout="wide")

# ─── SABITLER ────────────────────────────────────────────────────────────────
KEYS = ["LivingArea", "MainGroupDesc", "SubGroupDesc"]
MIN_PER_RANK = 10

# Varsayılan kolon isimleri
DEFAULT_COLS = {
    "sales_unit": "TY Sales Unit",
    "sales_value": "TY Sales Value TRY",
    "gross_profit": "TY Gross Profit TRY",
    "capacity": "CUR Total Store Capacity dm3",
    "price": "TY Sales Unit Price",
    "store": "StoreCode",
}

# Varsayılan ağırlıklar
DEFAULT_WEIGHTS = {
    "sales_unit": 0.2,
    "sales_value": 0.3,
    "gross_profit": 0.5,
    "perf_weighted": 0.7,
    "perf_price": 0.3,
}

# Rank → Cluster Code eşleştirmesi
RANK_TO_CODE = {
    1: "TOP1", 2: "TOP2", 3: "TOP3",
    4: "MID1", 5: "MID2", 6: "MID3",
    7: "ALL1", 8: "ALL2", 9: "ALL3",
}
CODE_TO_RANK = {v: k for k, v in RANK_TO_CODE.items()}

# Rank → (Kapasite, Performans) eşleştirmesi
RANK_TO_PAIR = {
    1: ("Büyük", "Hızlı"), 2: ("Büyük", "Orta"), 3: ("Büyük", "Yavaş"),
    4: ("Orta", "Hızlı"),  5: ("Orta", "Orta"),  6: ("Orta", "Yavaş"),
    7: ("Küçük", "Hızlı"), 8: ("Küçük", "Orta"), 9: ("Küçük", "Yavaş"),
}

# Key → Detay eşleştirmesi
KEY_TO_DETAIL = {
    "BüyükHızlı": "TOP1-Büyük-Hızlı", "BüyükOrta": "TOP2-Büyük-Orta", "BüyükYavaş": "TOP3-Büyük-Yavaş",
    "OrtaHızlı": "MID1-Orta-Hızlı",   "OrtaOrta": "MID2-Orta-Orta",   "OrtaYavaş": "MID3-Orta-Yavaş",
    "KüçükHızlı": "ALL1-Küçük-Hızlı", "KüçükOrta": "ALL2-Küçük-Orta", "KüçükYavaş": "ALL3-Küçük-Yavaş",
}

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2F5496 50%, #4a90d9 100%);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(15,23,42,0.25);
    }
    .app-header h1 {
        font-size: 1.65rem; font-weight: 700;
        color: #fff; margin: 0;
        letter-spacing: 0.5px;
    }
    .app-header h1 span { color: #60a5fa; }
    .app-header .sub {
        font-size: 0.82rem; color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 1.5px; text-transform: uppercase;
    }

    .section-header {
        background: #2F5496;
        color: white;
        padding: 10px 14px;
        font-weight: bold;
        font-size: 13px;
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

    .metric-box {
        background: #f0f4ff;
        border: 1px solid #c5d5f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-box .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2F5496;
    }
    .metric-box .label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
    }

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
</style>
""", unsafe_allow_html=True)


# ─── CORE FUNCTIONS ─────────────────────────────────────────────────────────

def safe_minmax(s: pd.Series) -> pd.Series:
    """Min-Max normalizasyon (0-1 arası)"""
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series([np.nan] * len(s), index=s.index)
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def kmeans_per_group(data: pd.DataFrame, weighted: pd.Series, price_sc: pd.Series,
                     cap_sc: pd.Series, keys: list, perf_w: float = 0.7,
                     random_state: int = 42, n_init: int = 20):
    """
    Her grup için ayrı K-Means:
    - Performans: 2D [weighted, price_sc] → Yavaş/Orta/Hızlı
    - Kapasite: 1D [cap_sc] → Küçük/Orta/Büyük
    """
    perf = pd.Series(index=data.index, dtype=object)
    cap = pd.Series(index=data.index, dtype=object)
    order_comp = pd.Series(index=data.index, dtype=float)

    for _, grp in data.groupby(keys):
        idx = grp.index

        # ─── PERFORMANS K-MEANS ───
        X = np.c_[weighted.loc[idx].values, price_sc.loc[idx].values]
        mask = np.isfinite(X).all(axis=1)

        if mask.sum() >= 3:
            km = KMeans(n_clusters=3, random_state=random_state, n_init=n_init)
            lab = km.fit_predict(X[mask])

            # Composite score
            comp = perf_w * weighted.loc[idx[mask]] + (1 - perf_w) * price_sc.loc[idx[mask]]
            centers = [(k, float(np.nanmean(comp.values[lab == k]))) for k in np.unique(lab)]
            order = [k for k, _ in sorted(centers, key=lambda t: t[1])]
            m = {order[0]: "Yavaş", order[1]: "Orta", order[2]: "Hızlı"}

            out = np.array(["Orta"] * len(idx), dtype=object)
            out[mask] = np.vectorize(lambda a: m.get(a, "Orta"))(lab)

            perf.loc[idx] = out
            order_comp.loc[idx[mask]] = comp.values
        else:
            # Fallback: quantile
            comp_all = perf_w * weighted.loc[idx] + (1 - perf_w) * price_sc.loc[idx]
            order_comp.loc[idx] = comp_all.values
            try:
                q = pd.qcut(comp_all.rank(method="first"), 3, labels=["Yavaş", "Orta", "Hızlı"])
                perf.loc[idx] = q.astype(str).values
            except Exception:
                perf.loc[idx] = "Orta"

        # ─── KAPASİTE K-MEANS ───
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
            except Exception:
                cap.loc[idx] = "Orta"

    return perf, cap, order_comp


def assign_cluster_code(cap: pd.Series, perf: pd.Series) -> tuple:
    """Kapasite + Performans → Cluster Code (TOP1-9, MID1-9, ALL1-9)"""

    # Key oluştur
    concat_key = (cap.astype(str) + perf.astype(str)).replace("nannan", "")

    # Pair → Code eşleştirmesi
    pair_to_code = {
        ("Büyük", "Hızlı"): "TOP1", ("Büyük", "Orta"): "TOP2", ("Büyük", "Yavaş"): "TOP3",
        ("Orta", "Hızlı"): "MID1",  ("Orta", "Orta"): "MID2",  ("Orta", "Yavaş"): "MID3",
        ("Küçük", "Hızlı"): "ALL1", ("Küçük", "Orta"): "ALL2", ("Küçük", "Yavaş"): "ALL3",
    }

    # Code ve Detay hesapla
    code = pd.Series([pair_to_code.get((c, p), np.nan) for c, p in zip(cap, perf)], index=cap.index)
    detail = concat_key.map(lambda k: KEY_TO_DETAIL.get(k, np.nan))

    return code, detail, concat_key


def enforce_min10_balancing(df: pd.DataFrame, keys: list, score_col: str = "_order_comp",
                            cap_sc_col: str = "Capacity_sc", min_per_rank: int = 10):
    """
    Min-10 Balancing: Her rank için minimum mağaza sayısı garantisi.
    Komşu rank'lardan sınır mağazaları kaydırarak dengeleme yapar.
    """

    # Rank hesapla
    base_rank = df["CO_PilotCluster"].map(lambda c: CODE_TO_RANK.get(c, np.nan))

    # Combined score
    score_med = df[score_col].median() if df[score_col].notna().any() else 0
    cap_med = df[cap_sc_col].median() if df[cap_sc_col].notna().any() else 0
    combined_score = (df[score_col].fillna(score_med) + df[cap_sc_col].fillna(cap_med)) / 2

    adj_rank = base_rank.copy()
    move_log = []

    for gk, grp in df.groupby(keys):
        idx = grp.index
        n = len(idx)

        # 9 rank × min = minimum gerekli
        if n < min_per_rank * 9:
            continue

        ranks = adj_rank.loc[idx].copy()
        scores = combined_score.loc[idx].copy()

        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            counts = ranks.value_counts()

            # Eksik rank'lar
            deficits = {r: min_per_rank - int(counts.get(r, 0))
                        for r in range(1, 10)
                        if int(counts.get(r, 0)) < min_per_rank}

            if not deficits:
                break

            # Fazla rank'lar
            donors = {r: int(counts.get(r, 0)) - min_per_rank
                      for r in range(1, 10)
                      if int(counts.get(r, 0)) > min_per_rank}

            if not donors:
                break

            # En çok eksiği olan hedef rank
            r_target = sorted(deficits.items(), key=lambda t: t[1], reverse=True)[0][0]
            need = deficits[r_target]

            # En yakın donor rank
            donor_candidates = sorted(donors.keys(), key=lambda r: (abs(r - r_target), r))
            if not donor_candidates:
                break

            r_donor = donor_candidates[0]
            take = min(need, donors[r_donor])

            donor_idx = ranks[ranks == r_donor].index

            # Sınır seçimi
            if r_donor < r_target:
                # İyi → Kötü: düşük skor taşınır
                pick = scores.loc[donor_idx].sort_values(ascending=True).head(take).index
            else:
                # Kötü → İyi: yüksek skor taşınır
                pick = scores.loc[donor_idx].sort_values(ascending=False).head(take).index

            ranks.loc[pick] = r_target

            move_log.append({
                "LivingArea": gk[0] if len(gk) > 0 else "",
                "MainGroupDesc": gk[1] if len(gk) > 1 else "",
                "SubGroupDesc": gk[2] if len(gk) > 2 else "",
                "from_rank": r_donor,
                "to_rank": r_target,
                "moved_count": len(pick)
            })

        adj_rank.loc[idx] = ranks

    # Yeni değerleri hesapla
    adj_code = adj_rank.map(lambda r: RANK_TO_CODE.get(int(r), np.nan) if pd.notna(r) else np.nan)
    adj_pair = adj_rank.map(lambda r: RANK_TO_PAIR.get(int(r), (np.nan, np.nan)) if pd.notna(r) else (np.nan, np.nan))
    adj_cap = adj_pair.map(lambda t: t[0])
    adj_perf = adj_pair.map(lambda t: t[1])

    adj_key = (adj_cap.astype(str) + adj_perf.astype(str)).replace("nannan", "")
    adj_detail = adj_key.map(lambda k: KEY_TO_DETAIL.get(k, np.nan))

    return adj_perf, adj_cap, adj_code, adj_detail, pd.DataFrame(move_log)


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    # Session state
    for key in ['data_df', 'results_df', 'move_log', 'config']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Header ──
    st.markdown("""
    <div class="app-header">
        <h1>🎯 CoPilot <span>Cluster</span> V3</h1>
        <div class="sub">LivingArea · MainGroup · SubGroup | Weighted K-Means + Min-10 Balancing</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2.5])

    # ══════════════════════════════════════════════════════════════════════════
    # SOL PANEL — INPUT
    # ══════════════════════════════════════════════════════════════════════════
    with col_left:
        st.markdown('<div class="section-header">📁 Veri Yükleme</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Excel dosyası yükle", type=['xlsx', 'xls'],
                                    key='data_upload', label_visibility="collapsed")

        if uploaded:
            try:
                xls = pd.ExcelFile(uploaded, engine='openpyxl')
                sheets = xls.sheet_names

                selected_sheet = st.selectbox("📄 Sayfa seçin", options=sheets, key='sheet_select')

                if selected_sheet:
                    df = xls.parse(selected_sheet)
                    st.session_state.data_df = df
                    st.success(f"✅ {len(df):,} satır yüklendi")
            except Exception as e:
                st.error(f"❌ Dosya okuma hatası: {e}")

        if st.session_state.data_df is not None:
            df = st.session_state.data_df
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── GRUP KOLONLARI ───
            st.markdown('<div class="section-header">🏷️ Grup Kolonları (Keys)</div>', unsafe_allow_html=True)

            col_living = st.selectbox("LivingArea", options=all_cols,
                                      index=all_cols.index("LivingArea") if "LivingArea" in all_cols else 0,
                                      key='col_living')
            col_main = st.selectbox("MainGroupDesc", options=all_cols,
                                    index=all_cols.index("MainGroupDesc") if "MainGroupDesc" in all_cols else 0,
                                    key='col_main')
            col_sub = st.selectbox("SubGroupDesc", options=all_cols,
                                   index=all_cols.index("SubGroupDesc") if "SubGroupDesc" in all_cols else 0,
                                   key='col_sub')

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── METRİK KOLONLARI ───
            st.markdown('<div class="section-header">📊 Metrik Kolonları</div>', unsafe_allow_html=True)

            def find_col(default_name, cols):
                """Varsayılan kolon adını bul veya ilkini döndür"""
                matches = [c for c in cols if default_name.lower() in c.lower()]
                if matches:
                    return cols.index(matches[0])
                return 0

            col_sales_unit = st.selectbox("📦 Satış Adet (E)", options=numeric_cols,
                                          index=find_col("Sales Unit", numeric_cols),
                                          key='col_e')
            col_sales_value = st.selectbox("💰 Ciro (F)", options=numeric_cols,
                                           index=find_col("Sales Value", numeric_cols),
                                           key='col_f')
            col_gross_profit = st.selectbox("📈 Brüt Kar (G)", options=numeric_cols,
                                            index=find_col("Gross Profit", numeric_cols),
                                            key='col_g')
            col_capacity = st.selectbox("🏪 Kapasite (I)", options=numeric_cols,
                                        index=find_col("Capacity", numeric_cols),
                                        key='col_i')
            col_price = st.selectbox("🏷️ Fiyat (J)", options=numeric_cols,
                                     index=find_col("Price", numeric_cols),
                                     key='col_j')

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── AĞIRLIKLAR ───
            st.markdown('<div class="section-header">⚖️ Weighted Score Ağırlıkları</div>', unsafe_allow_html=True)

            w_unit = st.slider("Satış Adet", 0, 100, 20, key='w_unit')
            w_value = st.slider("Ciro", 0, 100, 30, key='w_value')
            w_profit = st.slider("Brüt Kar", 0, 100, 50, key='w_profit')

            total_w = w_unit + w_value + w_profit
            if total_w > 0:
                st.caption(f"Toplam: {total_w} → Normalize edilecek")
            else:
                st.warning("⚠️ En az bir ağırlık > 0 olmalı!")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── PERFORMANS AĞIRLIĞI ───
            st.markdown('<div class="section-header">🎯 Performans Composite</div>', unsafe_allow_html=True)

            perf_w = st.slider("Weighted vs Price (0.7 = %70 Weighted)", 0.0, 1.0, 0.7, 0.05, key='perf_w')
            st.caption(f"Composite = {perf_w:.0%} Weighted + {1-perf_w:.0%} Price")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── MIN-10 AYARI ───
            st.markdown('<div class="section-header-orange">📏 Min-10 Balancing</div>', unsafe_allow_html=True)

            enable_min10 = st.checkbox("Min-10 Balancing Uygula", value=True, key='enable_min10')
            min_per_rank = st.number_input("Rank başına min mağaza", 1, 50, 10, key='min_rank')

            st.markdown("<hr>", unsafe_allow_html=True)

            # ─── HESAPLA BUTONU ───
            if st.button("🚀 Hesapla", use_container_width=True, type="primary"):
                with st.spinner("Hesaplanıyor..."):
                    try:
                        data = df.copy()
                        keys = [col_living, col_main, col_sub]

                        # Numerik dönüşüm
                        for c in [col_sales_unit, col_sales_value, col_gross_profit, col_capacity, col_price]:
                            data[c] = pd.to_numeric(data[c], errors='coerce')

                        # Group Min-Max Scale
                        E_sc = data.groupby(keys)[col_sales_unit].transform(safe_minmax)
                        F_sc = data.groupby(keys)[col_sales_value].transform(safe_minmax)
                        G_sc = data.groupby(keys)[col_gross_profit].transform(safe_minmax)
                        J_sc = data.groupby(keys)[col_price].transform(safe_minmax)
                        I_sc = data.groupby(keys)[col_capacity].transform(safe_minmax)

                        # Weighted Score
                        total = w_unit + w_value + w_profit
                        if total > 0:
                            weighted = (w_unit/total) * E_sc + (w_value/total) * F_sc + (w_profit/total) * G_sc
                        else:
                            weighted = E_sc

                        # K-Means per group
                        perf, cap, order_comp = kmeans_per_group(
                            data, weighted, J_sc, I_sc, keys, perf_w=perf_w
                        )

                        # Cluster code
                        code, detail, concat_key = assign_cluster_code(cap, perf)

                        # Sonuçları ekle
                        data["Weighted_Score"] = weighted
                        data["Price_sc"] = J_sc
                        data["Capacity_sc"] = I_sc
                        data["_order_comp"] = order_comp
                        data["PerformansGrade"] = perf
                        data["KapasiteCluster"] = cap
                        data["CO_PilotCluster"] = code
                        data["CO_PilotCluster_Detay"] = detail

                        move_log = None

                        # Min-10 Balancing
                        if enable_min10:
                            adj_perf, adj_cap, adj_code, adj_detail, move_log = enforce_min10_balancing(
                                data, keys, "_order_comp", "Capacity_sc", min_per_rank
                            )
                            data["PerformansGrade_min10"] = adj_perf
                            data["KapasiteCluster_min10"] = adj_cap
                            data["CO_PilotCluster_min10"] = adj_code
                            data["CO_PilotCluster_Detay_min10"] = adj_detail

                        st.session_state.results_df = data
                        st.session_state.move_log = move_log
                        st.session_state.config = {
                            'keys': keys,
                            'col_sales_unit': col_sales_unit,
                            'col_sales_value': col_sales_value,
                            'col_gross_profit': col_gross_profit,
                            'col_capacity': col_capacity,
                            'col_price': col_price,
                            'enable_min10': enable_min10,
                        }

                        st.success("✅ Hesaplama tamamlandı!")

                    except Exception as e:
                        st.error(f"❌ Hata: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("📁 Excel dosyası yükleyin")

    # ══════════════════════════════════════════════════════════════════════════
    # SAĞ PANEL — SONUÇLAR
    # ══════════════════════════════════════════════════════════════════════════
    with col_right:
        st.markdown('<div class="section-header">📊 Sonuçlar</div>', unsafe_allow_html=True)

        if st.session_state.results_df is not None:
            results = st.session_state.results_df
            cfg = st.session_state.config

            # Hangi cluster kolonunu kullanacağız?
            cluster_col = "CO_PilotCluster_min10" if cfg['enable_min10'] and "CO_PilotCluster_min10" in results.columns else "CO_PilotCluster"
            cap_col = "KapasiteCluster_min10" if cfg['enable_min10'] and "KapasiteCluster_min10" in results.columns else "KapasiteCluster"
            perf_col = "PerformansGrade_min10" if cfg['enable_min10'] and "PerformansGrade_min10" in results.columns else "PerformansGrade"

            # ── Legend ──
            st.markdown("""
            <div class="legend-box">
                <b>Cluster Kodları:</b><br>
                <span class="badge-top">TOP1-3</span> Büyük Mağaza (Hızlı → Yavaş)<br>
                <span class="badge-mid">MID1-3</span> Orta Mağaza (Hızlı → Yavaş)<br>
                <span class="badge-all">ALL1-3</span> Küçük Mağaza (Hızlı → Yavaş)
            </div>
            """, unsafe_allow_html=True)

            # ── KPI ──
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Satır", f"{len(results):,}")
            c2.metric("Grup Sayısı", f"{results[cfg['keys']].drop_duplicates().shape[0]:,}")
            c3.metric("Cluster", f"{results[cluster_col].nunique()}")

            if st.session_state.move_log is not None and len(st.session_state.move_log) > 0:
                c4.metric("Min-10 Taşıma", f"{st.session_state.move_log['moved_count'].sum():,}")
            else:
                c4.metric("Min-10 Taşıma", "0")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Cluster Dağılımı ──
            st.markdown("**📊 Cluster Dağılımı**")

            cluster_counts = results[cluster_col].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']

            # Sıralama
            order = ['TOP1', 'TOP2', 'TOP3', 'MID1', 'MID2', 'MID3', 'ALL1', 'ALL2', 'ALL3']
            cluster_counts['sort_key'] = cluster_counts['Cluster'].apply(lambda x: order.index(x) if x in order else 99)
            cluster_counts = cluster_counts.sort_values('sort_key').drop('sort_key', axis=1)

            # Renk haritası
            color_map = {
                'TOP1': '#006100', 'TOP2': '#38761d', 'TOP3': '#6aa84f',
                'MID1': '#9C5700', 'MID2': '#b8860b', 'MID3': '#daa520',
                'ALL1': '#9C0006', 'ALL2': '#cc0000', 'ALL3': '#ea9999',
            }

            fig = px.bar(cluster_counts, x='Cluster', y='Count',
                         color='Cluster', color_discrete_map=color_map,
                         height=300)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # ── 3D Scatter ──
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**🎯 3D Scatter — Kapasite × Performans × Weighted Score**")

            fig3d = px.scatter_3d(
                results.sample(min(5000, len(results))),  # Performance için sample
                x='Capacity_sc',
                y='_order_comp',
                z='Weighted_Score',
                color=cluster_col,
                color_discrete_map=color_map,
                opacity=0.7,
                height=500
            )
            fig3d.update_traces(marker=dict(size=4))
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Kapasite (scaled)',
                    yaxis_title='Performans Composite',
                    zaxis_title='Weighted Score',
                ),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

            # ── Özet Tablo ──
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**📋 Cluster Özet Tablosu**")

            summary = results.groupby(cluster_col).agg({
                cfg['col_sales_unit']: 'sum',
                cfg['col_sales_value']: 'sum',
                cfg['col_gross_profit']: 'sum',
                'Weighted_Score': 'mean',
                cfg['keys'][0]: 'count'
            }).reset_index()
            summary.columns = ['Cluster', 'Satış Adet', 'Ciro', 'Brüt Kar', 'Avg Score', 'Satır']

            # Sıralama
            summary['sort_key'] = summary['Cluster'].apply(lambda x: order.index(x) if x in order else 99)
            summary = summary.sort_values('sort_key').drop('sort_key', axis=1)

            # Formatlama
            for col in ['Satış Adet', 'Ciro', 'Brüt Kar', 'Satır']:
                summary[col] = summary[col].apply(lambda x: f"{int(x):,}".replace(",", "."))
            summary['Avg Score'] = summary['Avg Score'].apply(lambda x: f"{x:.3f}")

            st.dataframe(summary, hide_index=True, use_container_width=True)

            # ── Min-10 Log ──
            if st.session_state.move_log is not None and len(st.session_state.move_log) > 0:
                with st.expander("📝 Min-10 Balancing Log", expanded=False):
                    st.dataframe(st.session_state.move_log, hide_index=True, use_container_width=True)

            # ── İndir ──
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**📥 İndir**")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results.to_excel(writer, index=False, sheet_name='CoPilot_V3')
                    if st.session_state.move_log is not None:
                        st.session_state.move_log.to_excel(writer, index=False, sheet_name='Min10_Log')

                st.download_button(
                    "📥 Excel İndir",
                    buffer.getvalue(),
                    f"copilot_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )

            with col_d2:
                csv = results.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 CSV İndir",
                    csv,
                    f"copilot_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    use_container_width=True
                )

            # ── Detaylı Tablo ──
            with st.expander("📋 Tüm Veriyi Göster", expanded=False):
                show_cols = cfg['keys'] + [
                    cfg['col_sales_unit'], cfg['col_sales_value'], cfg['col_gross_profit'],
                    'Weighted_Score', cap_col, perf_col, cluster_col
                ]
                show_cols = [c for c in show_cols if c in results.columns]
                st.dataframe(results[show_cols], height=400, use_container_width=True)

        else:
            st.info("👈 Sol panelden veri yükleyip hesaplayın")

            # Demo görsel
            st.markdown("**📊 Demo — Örnek Cluster Dağılımı**")
            demo_data = pd.DataFrame({
                'Cluster': ['TOP1', 'TOP2', 'TOP3', 'MID1', 'MID2', 'MID3', 'ALL1', 'ALL2', 'ALL3'],
                'Count': [150, 180, 120, 200, 250, 180, 300, 350, 270]
            })
            fig = px.bar(demo_data, x='Cluster', y='Count',
                         color='Cluster',
                         color_discrete_sequence=['#006100', '#38761d', '#6aa84f',
                                                  '#9C5700', '#b8860b', '#daa520',
                                                  '#9C0006', '#cc0000', '#ea9999'],
                         height=300)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
