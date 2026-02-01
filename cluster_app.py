"""
Cluster Analizi — Mağaza Kapasite + Ürün + Fiyat (3D)
Per-Kategori Gruplama | TOP-1-A Format
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime

st.set_page_config(page_title="Cluster Analizi 3D", page_icon="📊", layout="wide")

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-cluster { padding: 2.5rem 1.5rem 1rem 1.5rem; }

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


def kmeans_global(df, attribute_cols, n_clusters):
    """
    Global K-Means — kapasite gruplama (tüm mağazalar üzerinde bir kez).
    Döner: 1=düşük … n=yüksek sıralı cluster numaraları.
    """
    X = df[attribute_cols].fillna(df[attribute_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Ortalama değere göre küçükten büyüğe sırala
    means = {c: X.iloc[clusters == c].mean().mean() for c in range(n_clusters)}
    sorted_c = sorted(means.keys(), key=lambda x: means[x])
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def kmeans_per_category(df, kategori_col, metric_col, n_clusters, label_type='numeric'):
    """
    Kategori bazında ayrı K-Means.
        label_type = 'numeric' → 1, 2, 3  (düşük → yüksek)
        label_type = 'alpha'   → A, B, C  (düşük → yüksek)
    Her kategori kendi içinde bağımsız olarak gruplandırılır.
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
        sorted_c = sorted(means.keys(), key=lambda x: means[x])
        mapping = {old: new for new, old in enumerate(sorted_c)}
        sorted_clusters = np.array([mapping[c] for c in clusters])

        if label_type == 'numeric':
            result.loc[mask] = sorted_clusters + 1          # 1, 2, 3…
        else:
            labels = [chr(65 + i) for i in range(actual_clusters)]  # A, B, C…
            result.loc[mask] = [labels[c] for c in sorted_clusters]

    return result


def get_kapasite_label(grup_num, total):
    """Kapasite grup numarası → TOP / MID / ALL"""
    if total == 2:
        return 'TOP' if grup_num == total else 'ALL'
    elif total == 3:
        return {1: 'ALL', 2: 'MID', 3: 'TOP'}.get(grup_num, str(grup_num))
    else:
        if grup_num == total:
            return 'TOP'
        elif grup_num == 1:
            return 'ALL'
        else:
            return 'MID'


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # Session state init
    for key in ['kapasite_df', 'urun_df', 'kapasite_results', 'final_results', 'config']:
        if key not in st.session_state:
            st.session_state[key] = None

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
                kap_df['_Kap_Grup_Num'] = kmeans_global(kap_df, kap_attrs, kap_grup_sayisi)
                kap_df['Kapasite_Grubu'] = kap_df['_Kap_Grup_Num'].apply(
                    lambda x: get_kapasite_label(x, kap_grup_sayisi)
                )
                st.session_state.kapasite_results = kap_df

                # ── STEP 2: Ürün — PER KATEGORI gruplama ─────────────────────
                urun_df = st.session_state.urun_df.copy()

                urun_df['Urun_Grubu']  = kmeans_per_category(
                    urun_df, urun_kategori_col, urun_metric_col, urun_grup_sayisi, 'numeric'
                )
                urun_df['Fiyat_Grubu'] = kmeans_per_category(
                    urun_df, urun_kategori_col, urun_fiyat_col, fiyat_grup_sayisi, 'alpha'
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
            st.markdown("""
            <div class="legend-box">
                <b>Format → TOP-1-A</b><br>
                <b>TOP / MID / ALL</b> — Mağaza Kapasite grubu (global, ürün bağımsız)<br>
                <b>1 / 2 / 3</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Ürün Performans grubu (<i>her kategori içinde ayrı</i>)<br>
                <b>A / B / C</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Fiyat Seviyesi grubu&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<i>her kategori içinde ayrı</i>)<br>
                Tüm boyutlarda <b>düşük → yüksek</b> sıralama yapılır.
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Satır",  f"{len(filtered):,}")
            c2.metric("Mağaza",        f"{filtered[urun_magaza_col].nunique():,}")
            c3.metric("Ürün",          f"{filtered[urun_urun_col].nunique():,}")
            c4.metric("Kombine Grup",  f"{filtered['Kombine_Grup'].nunique()}")

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
                            'Kapasite_Grubu', 'Urun_Grubu', 'Fiyat_Grubu'],
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
            # KOMBINE GRUP TABLOSU
            # ══════════════════════════════════════════════════════════════════
            st.markdown("**📊 Kombine Grup Özeti**")

            combo = (filtered
                     .groupby(['Kapasite_Grubu', 'Urun_Grubu', 'Fiyat_Grubu', 'Kombine_Grup'])
                     .size()
                     .reset_index(name='Adet')
                     .sort_values(['Kapasite_Grubu', 'Urun_Grubu', 'Fiyat_Grubu']))

            total = len(filtered)
            html  = '<table class="result-table">'
            html += '<tr><th>Kombine</th><th>Kapasite</th><th>Ürün</th><th>Fiyat</th><th>Adet</th><th>%</th></tr>'

            for _, row in combo.iterrows():
                kap_g  = row['Kapasite_Grubu']
                badge  = ('badge-top' if kap_g == 'TOP'
                          else 'badge-mid' if kap_g == 'MID'
                          else 'badge-all')
                pct    = row['Adet'] / total * 100 if total else 0
                html  += (
                    f'<tr>'
                    f'<td><b>{row["Kombine_Grup"]}</b></td>'
                    f'<td><span class="{badge}">{kap_g}</span></td>'
                    f'<td>{row["Urun_Grubu"]}</td>'
                    f'<td>{row["Fiyat_Grubu"]}</td>'
                    f'<td>{row["Adet"]}</td>'
                    f'<td>{pct:.1f}%</td>'
                    f'</tr>'
                )
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)

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
                urun_fiyat_col, 'Fiyat_Grubu', 'Kombine_Grup'
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
            st.info("👈 Sol panelden verileri yükleyin ve 'Grupla ve Birleştir' butonuna tıklayın.")


if __name__ == "__main__":
    main()
