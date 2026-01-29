"""
Cluster Analizi - Her boyut için ayrı K-Means
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

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a2c7a 100%);
        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; text-align: center;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def cluster_dimension(values, n_clusters):
    """Tek boyut için K-Means"""
    X = values.fillna(values.mean()).values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Cluster'ları değere göre sırala (1=düşük, n=yüksek)
    means = {c: values[clusters == c].mean() for c in range(n_clusters)}
    sorted_c = sorted(means.keys(), key=lambda x: means[x])
    mapping = {old: new + 1 for new, old in enumerate(sorted_c)}

    return np.array([mapping[c] for c in clusters])


def main():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Cluster Analizi</h1>
        <p>Her boyut için ayrı gruplama</p>
    </div>
    """, unsafe_allow_html=True)

    # Session state init
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Dosya")
        uploaded = st.file_uploader("Excel/CSV Yükle", type=['xlsx', 'xls', 'csv'])

        if uploaded:
            st.session_state.df = load_data(uploaded)
            st.success(f"✅ {len(st.session_state.df)} satır")

    df = st.session_state.df

    if df is None:
        st.info("👈 Sol panelden dosya yükleyin")
        return

    # Kolon listesi
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("---")

    # Ana ayarlar - 3 kolon
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Neyi Grupla?")
        name_col = st.selectbox("Etiket Kolonu", options=all_cols, index=0)

    with col2:
        st.markdown("### 2️⃣ Boyutlar")
        available = [c for c in numeric_cols if c != name_col]

        boyut1 = st.selectbox("Boyut 1", options=["-- Seç --"] + available)
        boyut2 = st.selectbox("Boyut 2 (opsiyonel)", options=["-- Seç --"] + [c for c in available if c != boyut1])
        boyut3 = st.selectbox("Boyut 3 (opsiyonel)", options=["-- Seç --"] + [c for c in available if c not in [boyut1, boyut2]])

    with col3:
        st.markdown("### 3️⃣ Grup Sayıları")
        g1 = st.number_input("Boyut 1 Grup", min_value=2, max_value=15, value=3, disabled=(boyut1=="-- Seç --"))
        g2 = st.number_input("Boyut 2 Grup", min_value=2, max_value=15, value=3, disabled=(boyut2=="-- Seç --"))
        g3 = st.number_input("Boyut 3 Grup", min_value=2, max_value=15, value=3, disabled=(boyut3=="-- Seç --"))

    # Seçimleri topla
    boyutlar = []
    gruplar = []
    if boyut1 != "-- Seç --":
        boyutlar.append(boyut1)
        gruplar.append(int(g1))
    if boyut2 != "-- Seç --":
        boyutlar.append(boyut2)
        gruplar.append(int(g2))
    if boyut3 != "-- Seç --":
        boyutlar.append(boyut3)
        gruplar.append(int(g3))

    st.markdown("---")

    # Buton
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run = st.button("🚀 GRUPLA", use_container_width=True, type="primary", disabled=(len(boyutlar)==0))

    if run and len(boyutlar) > 0:
        with st.spinner("Gruplama yapılıyor..."):
            results = df.copy()

            for boyut, n_grup in zip(boyutlar, gruplar):
                results[f'{boyut}_Grup'] = cluster_dimension(df[boyut], n_grup)

            # Kombine grup
            grup_cols = [f'{b}_Grup' for b in boyutlar]
            if len(grup_cols) == 1:
                results['Kombine_Grup'] = results[grup_cols[0]].astype(str)
            else:
                results['Kombine_Grup'] = results[grup_cols].astype(str).agg('-'.join, axis=1)

            st.session_state.results = results
            st.session_state.boyutlar = boyutlar
            st.session_state.gruplar = gruplar
            st.session_state.name_col = name_col

    # Sonuçlar
    if st.session_state.results is not None:
        results = st.session_state.results
        boyutlar = st.session_state.boyutlar
        gruplar = st.session_state.gruplar
        name_col = st.session_state.name_col

        st.markdown("---")

        # Özet
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Kayıt", len(results))
        c2.metric("Boyut Sayısı", len(boyutlar))
        c3.metric("Benzersiz Grup", results['Kombine_Grup'].nunique())

        # Tablar
        tab1, tab2, tab3 = st.tabs(["📈 Grafik", "📊 Dağılım", "📋 Tablo"])

        with tab1:
            if len(boyutlar) == 1:
                fig = px.histogram(results, x=boyutlar[0], color=f'{boyutlar[0]}_Grup', barmode='overlay')
            elif len(boyutlar) == 2:
                fig = px.scatter(results, x=boyutlar[0], y=boyutlar[1], color='Kombine_Grup',
                               hover_data=[name_col], opacity=0.7)
                fig.update_traces(marker=dict(size=10))
            else:
                fig = px.scatter_3d(results, x=boyutlar[0], y=boyutlar[1], z=boyutlar[2],
                                   color='Kombine_Grup', hover_data=[name_col], opacity=0.7)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            for boyut, n_grup in zip(boyutlar, gruplar):
                st.markdown(f"**{boyut}** → {n_grup} grup")
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig = px.box(results, x=f'{boyut}_Grup', y=boyut, color=f'{boyut}_Grup')
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    stats = results.groupby(f'{boyut}_Grup')[boyut].agg(['count','mean','min','max']).round(2)
                    stats.columns = ['Adet','Ort','Min','Max']
                    st.dataframe(stats)
                st.markdown("---")

            # Kombine
            st.markdown("**Kombine Grup Dağılımı**")
            combo = results['Kombine_Grup'].value_counts().head(20)
            fig = px.bar(x=combo.index, y=combo.values, labels={'x':'Grup', 'y':'Adet'})
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Sütun sırası
            show_cols = [name_col] + boyutlar + [f'{b}_Grup' for b in boyutlar] + ['Kombine_Grup']
            other = [c for c in results.columns if c not in show_cols]
            show_cols += other

            st.dataframe(results[show_cols], height=500, use_container_width=True)

            # İndir
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results[show_cols].to_excel(writer, index=False, sheet_name='Veri')
                st.download_button("📥 Excel İndir", buffer.getvalue(),
                                 f"gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                 use_container_width=True)

            with col_d2:
                csv = results[show_cols].to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 CSV İndir", csv,
                                 f"gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                 use_container_width=True)


if __name__ == "__main__":
    main()
