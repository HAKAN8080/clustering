"""
Cluster Analizi - Mağaza ve Ürün Gruplama
"""

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

# CSS - Excel benzeri grid tasarım
st.markdown("""
<style>
    /* Genel ayarlar */
    .block-container { padding: 1rem 2rem; }

    /* Başlık stilleri */
    .section-header {
        background: #C00000;
        color: white;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
        margin: 0;
        border: 1px solid #999;
    }
    .section-header-blue {
        background: #2F5496;
        color: white;
        padding: 10px 12px;
        font-weight: bold;
        font-size: 14px;
        margin: 0;
        border: 1px solid #999;
        text-align: center;
    }

    /* Grid form stili */
    .form-grid {
        display: grid;
        grid-template-columns: 180px 120px;
        border: 1px solid #999;
        background: white;
    }
    .form-label {
        padding: 8px 10px;
        border: 1px solid #D9D9D9;
        background: #F2F2F2;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .form-value {
        padding: 4px 8px;
        border: 1px solid #D9D9D9;
        background: white;
        font-size: 13px;
    }

    /* Sonuç tablosu */
    .result-table {
        border-collapse: collapse;
        font-size: 13px;
        width: 100%;
    }
    .result-table th {
        background: #2F5496;
        color: white;
        padding: 8px 12px;
        border: 1px solid #999;
        text-align: center;
        font-weight: 600;
    }
    .result-table td {
        padding: 6px 12px;
        border: 1px solid #D9D9D9;
        text-align: center;
        background: white;
    }
    .result-table tr:nth-child(even) td {
        background: #F2F2F2;
    }

    /* Grup renkleri */
    .grup-top { background: #C6EFCE !important; color: #006100; font-weight: bold; }
    .grup-mid { background: #FFEB9C !important; color: #9C5700; font-weight: bold; }
    .grup-all { background: #FFC7CE !important; color: #9C0006; font-weight: bold; }

    /* Buton stili */
    .stButton > button {
        background: #2F5496;
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: #1e3a5f;
        color: white;
    }

    /* Selectbox ve input küçültme */
    .stSelectbox, .stMultiSelect, .stNumberInput {
        margin-bottom: 0 !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        min-height: 35px !important;
    }

    /* Divider */
    hr { margin: 10px 0; border-color: #D9D9D9; }
</style>
""", unsafe_allow_html=True)


def load_data(uploaded_file):
    """Excel veya CSV dosyası yükle"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def get_magaza_grup_name(grup_num, total_groups):
    """Mağaza grubu için isim: TOP, MID, ALL"""
    if total_groups == 3:
        names = {1: 'ALL', 2: 'MID', 3: 'TOP'}
        return names.get(grup_num, str(grup_num))
    elif total_groups == 2:
        names = {1: 'ALL', 2: 'TOP'}
        return names.get(grup_num, str(grup_num))
    else:
        if grup_num == total_groups:
            return 'TOP'
        elif grup_num == 1:
            return 'ALL'
        else:
            return 'MID'


def multi_dimension_cluster(df, attribute_cols, n_clusters):
    """Çok boyutlu K-Means clustering"""
    X = df[attribute_cols].fillna(df[attribute_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Cluster'ları ortalama değere göre sırala
    means = {}
    for c in range(n_clusters):
        mask = clusters == c
        means[c] = df[attribute_cols].iloc[mask].mean().mean()

    sorted_c = sorted(means.keys(), key=lambda x: means[x])
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def create_download_excel(df, sheet_name='Veri'):
    """DataFrame'i Excel buffer'a çevir"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def create_detailed_report(magaza_results, birlesik_results, magaza_label_col,
                           urun_magaza_col, urun_urun_col, magaza_attrs, urun_attrs):
    """Detaylı rapor Excel"""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Özet
        ozet = pd.DataFrame([
            {'Metrik': 'Rapor Tarihi', 'Değer': datetime.now().strftime('%Y-%m-%d %H:%M')},
            {'Metrik': 'Toplam Mağaza', 'Değer': len(magaza_results)},
            {'Metrik': 'Toplam Kayıt', 'Değer': len(birlesik_results)},
            {'Metrik': 'Benzersiz Ürün', 'Değer': birlesik_results[urun_urun_col].nunique()},
        ])
        ozet.to_excel(writer, index=False, sheet_name='Özet')

        # Mağaza Grupları
        magaza_results.to_excel(writer, index=False, sheet_name='Mağaza Grupları')

        # Birleşik
        birlesik_results.to_excel(writer, index=False, sheet_name='Birleşik Sonuç')

        # Mağaza İstatistik
        m_stats = []
        for grup in sorted(magaza_results['Magaza_Grup'].unique()):
            g_df = magaza_results[magaza_results['Magaza_Grup'] == grup]
            row = {'Grup': grup, 'Grup Adı': g_df['Magaza_Grup_Adi'].iloc[0], 'Adet': len(g_df)}
            for col in magaza_attrs:
                row[f'{col}_Ort'] = g_df[col].mean()
                row[f'{col}_Min'] = g_df[col].min()
                row[f'{col}_Max'] = g_df[col].max()
            m_stats.append(row)
        pd.DataFrame(m_stats).to_excel(writer, index=False, sheet_name='Mağaza İstatistik')

        # Ürün İstatistik
        u_stats = []
        for grup in sorted(birlesik_results['Urun_Grup'].unique()):
            g_df = birlesik_results[birlesik_results['Urun_Grup'] == grup]
            row = {'Ürün Grubu': grup, 'Adet': len(g_df)}
            for col in urun_attrs:
                row[f'{col}_Ort'] = g_df[col].mean()
                row[f'{col}_Min'] = g_df[col].min()
                row[f'{col}_Max'] = g_df[col].max()
            u_stats.append(row)
        pd.DataFrame(u_stats).to_excel(writer, index=False, sheet_name='Ürün İstatistik')

        # Çapraz Tablo
        cross = pd.crosstab(birlesik_results['Magaza_Grup_Adi'], birlesik_results['Urun_Grup'])
        cross.to_excel(writer, sheet_name='Çapraz Tablo')

    return buffer.getvalue()


def render_stats_table(df, grup_col, grup_adi_col=None):
    """İstatistik tablosu HTML"""
    if grup_adi_col and grup_adi_col in df.columns:
        counts = df.groupby([grup_col, grup_adi_col]).size().reset_index(name='Adet')
        counts = counts.sort_values(grup_col)
    else:
        counts = df[grup_col].value_counts().sort_index().reset_index()
        counts.columns = [grup_col, 'Adet']

    html = '<table class="result-table"><tr><th>Grup</th><th>Adet</th></tr>'
    for _, row in counts.iterrows():
        grup_adi = row.get(grup_adi_col, row[grup_col]) if grup_adi_col else row[grup_col]
        html += f'<tr><td>{grup_adi}</td><td>{row["Adet"]}</td></tr>'
    html += '</table>'
    return html


def main():
    # Session state
    if 'magaza_df' not in st.session_state:
        st.session_state.magaza_df = None
    if 'magaza_results' not in st.session_state:
        st.session_state.magaza_results = None
    if 'urun_df' not in st.session_state:
        st.session_state.urun_df = None
    if 'birlesik_results' not in st.session_state:
        st.session_state.birlesik_results = None

    # Ana layout: Sol (form) | Sağ (sonuçlar)
    col_left, col_right = st.columns([1, 2])

    # ==================== SOL PANEL - FORM ====================
    with col_left:
        # MAĞAZA GRUPLAMA
        st.markdown('<div class="section-header">Mağaza Gruplama</div>', unsafe_allow_html=True)

        uploaded_magaza = st.file_uploader("Veri yükleme", type=['xlsx', 'xls', 'csv'],
                                           key='magaza_upload', label_visibility="collapsed")
        if uploaded_magaza:
            st.session_state.magaza_df = load_data(uploaded_magaza)

        if st.session_state.magaza_df is not None:
            df_m = st.session_state.magaza_df
            all_cols_m = df_m.columns.tolist()
            numeric_cols_m = df_m.select_dtypes(include=[np.number]).columns.tolist()

            magaza_label = st.selectbox("🏷️ Mağaza ID/İsim Kolonu", options=all_cols_m, key='m_label')
            available_m = [c for c in numeric_cols_m if c != magaza_label]
            magaza_attrs = st.multiselect("📊 Attribute Kolonları", options=available_m,
                                          default=available_m[:min(2, len(available_m))], key='m_attrs')
            magaza_grup_sayisi = st.number_input("Grup Sayısı", min_value=2, max_value=20, value=3, key='m_grup')
        else:
            st.caption("📁 Mağaza verisi yükleyin")
            magaza_label = None
            magaza_attrs = []
            magaza_grup_sayisi = 3

        st.markdown("<hr>", unsafe_allow_html=True)

        # ÜRÜN GRUPLAMA
        st.markdown('<div class="section-header">Ürün Gruplama</div>', unsafe_allow_html=True)

        uploaded_urun = st.file_uploader("Veri yükleme", type=['xlsx', 'xls', 'csv'],
                                         key='urun_upload', label_visibility="collapsed")
        if uploaded_urun:
            st.session_state.urun_df = load_data(uploaded_urun)

        if st.session_state.urun_df is not None:
            df_u = st.session_state.urun_df
            all_cols_u = df_u.columns.tolist()
            numeric_cols_u = df_u.select_dtypes(include=[np.number]).columns.tolist()

            urun_magaza_col = st.selectbox("🏪 Mağaza Kolonu", options=all_cols_u, key='u_magaza')
            urun_urun_col = st.selectbox("📦 Ürün Kolonu",
                                         options=[c for c in all_cols_u if c != urun_magaza_col], key='u_urun')
            available_u = [c for c in numeric_cols_u if c not in [urun_magaza_col, urun_urun_col]]
            urun_attrs = st.multiselect("📊 Attribute Kolonları", options=available_u,
                                        default=available_u[:min(2, len(available_u))], key='u_attrs')
            urun_grup_sayisi = st.number_input("Grup Sayısı", min_value=2, max_value=20, value=3, key='u_grup')
        else:
            st.caption("📁 Ürün verisi yükleyin")
            urun_magaza_col = None
            urun_urun_col = None
            urun_attrs = []
            urun_grup_sayisi = 3

        st.markdown("<hr>", unsafe_allow_html=True)

        # GRUPLA BUTONU
        btn_disabled = (st.session_state.magaza_df is None or
                       st.session_state.urun_df is None or
                       len(magaza_attrs) == 0 or
                       len(urun_attrs) == 0)

        if st.button("🚀 Grupla ve Birleştir", disabled=btn_disabled, use_container_width=True, type="primary"):
            with st.spinner("Gruplama yapılıyor..."):
                # Mağaza gruplama
                results_m = st.session_state.magaza_df.copy()
                results_m['Magaza_Grup'] = multi_dimension_cluster(results_m, magaza_attrs, magaza_grup_sayisi)
                results_m['Magaza_Grup_Adi'] = results_m['Magaza_Grup'].apply(
                    lambda x: get_magaza_grup_name(x, magaza_grup_sayisi)
                )
                st.session_state.magaza_results = results_m
                st.session_state.magaza_label_col = magaza_label
                st.session_state.magaza_attrs_used = magaza_attrs

                # Ürün gruplama
                results_u = st.session_state.urun_df.copy()
                results_u['Urun_Grup'] = multi_dimension_cluster(results_u, urun_attrs, urun_grup_sayisi)

                # Birleştir
                magaza_gruplar = results_m[[magaza_label, 'Magaza_Grup', 'Magaza_Grup_Adi']].copy()
                magaza_gruplar.columns = [urun_magaza_col, 'Magaza_Grup', 'Magaza_Grup_Adi']
                results_u = results_u.merge(magaza_gruplar, on=urun_magaza_col, how='left')
                results_u['Magaza_Grup'] = results_u['Magaza_Grup'].fillna(0).astype(int)
                results_u['Magaza_Grup_Adi'] = results_u['Magaza_Grup_Adi'].fillna('?')

                # Kombine grup
                results_u['Kombine_Grup'] = results_u['Magaza_Grup_Adi'] + '-' + results_u['Urun_Grup'].astype(str)

                st.session_state.birlesik_results = results_u
                st.session_state.urun_magaza_col = urun_magaza_col
                st.session_state.urun_urun_col = urun_urun_col
                st.session_state.urun_attrs_used = urun_attrs

    # ==================== SAĞ PANEL - SONUÇLAR ====================
    with col_right:
        st.markdown('<div class="section-header-blue">Sonuç İstatistikleri</div>', unsafe_allow_html=True)

        if st.session_state.birlesik_results is not None:
            results = st.session_state.birlesik_results
            urun_magaza_col = st.session_state.urun_magaza_col
            urun_urun_col = st.session_state.urun_urun_col
            urun_attrs = st.session_state.urun_attrs_used
            magaza_attrs = st.session_state.magaza_attrs_used

            # Sonuç özeti
            st.markdown("**Sonuç**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Toplam Satır", f"{len(results):,}")
            c2.metric("Mağaza", f"{results[urun_magaza_col].nunique():,}")
            c3.metric("Ürün", f"{results[urun_urun_col].nunique():,}")
            c4.metric("Kombine Grup", f"{results['Kombine_Grup'].nunique()}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # 3D GRAFİK
            st.markdown("**3D Grafik - Kombine / Mağaza / Ürün** (seçilebilir, default: Kombine)")

            grafik_tipi = st.radio("Renklendirme:",
                                   ["Kombine", "Mağaza Grubu", "Ürün Grubu"],
                                   horizontal=True, label_visibility="collapsed")

            if len(urun_attrs) >= 2:
                # Renklendirme seçimi
                if grafik_tipi == "Kombine":
                    color_col = 'Kombine_Grup'
                elif grafik_tipi == "Mağaza Grubu":
                    color_col = 'Magaza_Grup_Adi'
                else:
                    color_col = 'Urun_Grup'

                # 3D veya 2D grafik
                if len(urun_attrs) >= 3:
                    fig = px.scatter_3d(
                        results,
                        x=urun_attrs[0],
                        y=urun_attrs[1],
                        z=urun_attrs[2],
                        color=color_col,
                        hover_data=[urun_magaza_col, urun_urun_col],
                        opacity=0.7,
                        height=450
                    )
                    fig.update_traces(marker=dict(size=5))
                else:
                    fig = px.scatter(
                        results,
                        x=urun_attrs[0],
                        y=urun_attrs[1],
                        color=color_col,
                        hover_data=[urun_magaza_col, urun_urun_col],
                        opacity=0.7,
                        height=400
                    )
                    fig.update_traces(marker=dict(size=8))

                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # MAĞAZA KAPASİTE GRUP - Box Plot ve Tablo
            st.markdown("**Mağaza Kapasite Grup**")

            col_box, col_table = st.columns([2, 1])

            with col_box:
                # Box plot
                magaza_results = st.session_state.magaza_results
                if len(magaza_attrs) > 0:
                    attr_for_box = magaza_attrs[0]
                    st.caption(f"#{attr_for_box} → {magaza_grup_sayisi} grup")

                    fig_box = px.box(
                        magaza_results,
                        x='Magaza_Grup',
                        y=attr_for_box,
                        color='Magaza_Grup',
                        labels={'Magaza_Grup': '#Magaza_Grup'},
                        height=300
                    )
                    fig_box.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="#Magaza_Grup"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            with col_table:
                # İstatistik tablosu
                st.markdown(render_stats_table(magaza_results, 'Magaza_Grup', 'Magaza_Grup_Adi'),
                           unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ÜRÜN GRUP - Box Plot ve Tablo
            st.markdown("**Ürün Grup**")

            col_box2, col_table2 = st.columns([2, 1])

            with col_box2:
                if len(urun_attrs) > 0:
                    attr_for_box2 = urun_attrs[0]
                    st.caption(f"#{attr_for_box2} → {urun_grup_sayisi} grup")

                    fig_box2 = px.box(
                        results,
                        x='Urun_Grup',
                        y=attr_for_box2,
                        color='Urun_Grup',
                        height=300
                    )
                    fig_box2.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="#Urun_Grup"
                    )
                    st.plotly_chart(fig_box2, use_container_width=True)

            with col_table2:
                urun_counts = results['Urun_Grup'].value_counts().sort_index()
                html = '<table class="result-table"><tr><th>Ürün Grubu</th><th>Adet</th></tr>'
                for grup, adet in urun_counts.items():
                    html += f'<tr><td>{grup}</td><td>{adet}</td></tr>'
                html += '</table>'
                st.markdown(html, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # İNDİRME BUTONLARI
            st.markdown("**📥 İndir**")
            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                excel_data = create_download_excel(results, 'Sonuc')
                st.download_button(
                    "📥 Sonuç Excel",
                    excel_data,
                    f"sonuc_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )

            with col_d2:
                excel_magaza = create_download_excel(st.session_state.magaza_results, 'Magaza')
                st.download_button(
                    "📥 Mağaza Grupları",
                    excel_magaza,
                    f"magaza_gruplari_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )

            with col_d3:
                try:
                    detailed = create_detailed_report(
                        st.session_state.magaza_results,
                        results,
                        st.session_state.magaza_label_col,
                        urun_magaza_col,
                        urun_urun_col,
                        magaza_attrs,
                        urun_attrs
                    )
                    st.download_button(
                        "📥 Detaylı Rapor",
                        detailed,
                        f"detayli_rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        use_container_width=True,
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")

            # VERİ TABLOSU
            with st.expander("📋 Tüm Veriyi Göster"):
                show_cols = [urun_magaza_col, 'Magaza_Grup_Adi', urun_urun_col, 'Urun_Grup', 'Kombine_Grup'] + urun_attrs
                st.dataframe(results[show_cols], height=400, use_container_width=True)

        else:
            st.info("👈 Sol panelden verileri yükleyin ve 'Grupla ve Birleştir' butonuna tıklayın.")


if __name__ == "__main__":
    main()
