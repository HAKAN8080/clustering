"""
Mağaza Gruplama Analizi - Çok Boyutlu K-Means Clustering
Her boyut için ayrı gruplama yapılır
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime

# Sayfa ayarları
st.set_page_config(
    page_title="Cluster Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a2c7a 50%, #1e3a5f 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .main-header h1 { color: white; font-size: 2rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); margin-top: 0.3rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }

    .metric-card.blue { background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); }
    .metric-card.purple { background: linear-gradient(135deg, #4a2c7a 0%, #7b4397 100%); }
    .metric-card.teal { background: linear-gradient(135deg, #0d7377 0%, #14919b 100%); }

    .metric-value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; }

    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #1e3a5f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cluster renkleri
CLUSTER_COLORS = [
    '#667eea', '#764ba2', '#0d7377', '#f093fb', '#f5576c',
    '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140',
    '#a8edea', '#fed6e3', '#d299c2', '#fef9d7', '#d4fc79'
]


def load_data(uploaded_file):
    """Dosya yükle"""
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    return None


def get_numeric_columns(df):
    """Sayısal sütunları al"""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def cluster_single_dimension(values, n_clusters):
    """Tek boyut için K-Means uygula"""
    X = values.fillna(values.mean()).values.reshape(-1, 1)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Cluster'ları değer ortalamasına göre sırala (düşükten yükseğe)
    cluster_means = {}
    for c in range(n_clusters):
        cluster_means[c] = values[clusters == c].mean()

    # Sıralama: en düşük ortalama = 1, en yüksek = n_clusters
    sorted_clusters = sorted(cluster_means.keys(), key=lambda x: cluster_means[x])
    cluster_map = {old: new + 1 for new, old in enumerate(sorted_clusters)}

    # Yeni cluster numaraları
    new_clusters = np.array([cluster_map[c] for c in clusters])

    return new_clusters, kmeans


def render_metric_card(label, value, color_class=""):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


# Ana uygulama
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Cluster Analizi</h1>
        <p>Her boyut için ayrı gruplama</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Dosya Yükle")

        uploaded_file = st.file_uploader(
            "Excel veya CSV",
            type=['xlsx', 'xls', 'csv'],
            help="Verilerinizi içeren dosyayı yükleyin"
        )

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success(f"✅ {len(df)} satır yüklendi")
        else:
            df = None

        if df is not None:
            st.markdown("---")
            numeric_cols = get_numeric_columns(df)
            all_cols = df.columns.tolist()

            # 1. Neyi grupla
            st.markdown("### 1️⃣ Neyi Grupla?")
            name_col = st.selectbox(
                "Gruplanacak Kolon",
                options=all_cols,
                index=0,
                help="Gruplamak istediğiniz öğeleri içeren kolon"
            )

            # 2. Boyutlar ve her birinin grup sayısı
            st.markdown("### 2️⃣ Boyutlar ve Grup Sayıları")

            available_metrics = [col for col in numeric_cols if col != name_col]

            # Boyut 1
            st.markdown("**Boyut 1**")
            boyut1 = st.selectbox("Kolon", options=["Seçiniz..."] + available_metrics, key="b1")
            grup1 = st.slider("Grup Sayısı", 2, 10, 3, key="g1") if boyut1 != "Seçiniz..." else 3

            # Boyut 2
            st.markdown("**Boyut 2 (Opsiyonel)**")
            remaining2 = [c for c in available_metrics if c != boyut1]
            boyut2 = st.selectbox("Kolon", options=["Seçiniz..."] + remaining2, key="b2")
            grup2 = st.slider("Grup Sayısı", 2, 10, 3, key="g2") if boyut2 != "Seçiniz..." else 3

            # Boyut 3
            st.markdown("**Boyut 3 (Opsiyonel)**")
            remaining3 = [c for c in remaining2 if c != boyut2]
            boyut3 = st.selectbox("Kolon", options=["Seçiniz..."] + remaining3, key="b3")
            grup3 = st.slider("Grup Sayısı", 2, 10, 3, key="g3") if boyut3 != "Seçiniz..." else 3

            st.markdown("---")
            run_analysis = st.button("🚀 Grupla", use_container_width=True)
        else:
            run_analysis = False
            name_col = None
            boyut1 = boyut2 = boyut3 = "Seçiniz..."
            grup1 = grup2 = grup3 = 3

    # Ana içerik
    if df is None:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Hoş Geldiniz!</h3>
            <p>Başlamak için sol panelden dosyanızı yükleyin.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Seçilen boyutları topla
    boyutlar = []
    grup_sayilari = []

    if boyut1 != "Seçiniz...":
        boyutlar.append(boyut1)
        grup_sayilari.append(grup1)
    if boyut2 != "Seçiniz...":
        boyutlar.append(boyut2)
        grup_sayilari.append(grup2)
    if boyut3 != "Seçiniz...":
        boyutlar.append(boyut3)
        grup_sayilari.append(grup3)

    if len(boyutlar) == 0:
        st.warning("⚠️ Lütfen en az 1 boyut seçin")
        return

    # Session state
    if 'results' not in st.session_state:
        st.session_state.results = None

    if run_analysis:
        with st.spinner("Gruplama yapılıyor..."):
            results = df.copy()

            # Her boyut için ayrı clustering
            for i, (boyut, n_grup) in enumerate(zip(boyutlar, grup_sayilari)):
                clusters, _ = cluster_single_dimension(df[boyut], n_grup)
                results[f'{boyut}_Grup'] = clusters

            # Kombine grup oluştur
            if len(boyutlar) == 1:
                results['Kombine_Grup'] = results[f'{boyutlar[0]}_Grup'].astype(str)
            else:
                grup_cols = [f'{b}_Grup' for b in boyutlar]
                results['Kombine_Grup'] = results[grup_cols].astype(str).agg('-'.join, axis=1)

            st.session_state.results = results
            st.session_state.boyutlar = boyutlar
            st.session_state.grup_sayilari = grup_sayilari
            st.session_state.name_col = name_col

    if st.session_state.results is not None:
        results = st.session_state.results
        boyutlar = st.session_state.boyutlar
        grup_sayilari = st.session_state.grup_sayilari
        name_col = st.session_state.name_col

        # Üst metrikler
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(render_metric_card("Toplam Kayıt", len(results), "blue"), unsafe_allow_html=True)

        with col2:
            st.markdown(render_metric_card("Boyut Sayısı", len(boyutlar), "purple"), unsafe_allow_html=True)

        with col3:
            unique_combos = results['Kombine_Grup'].nunique()
            st.markdown(render_metric_card("Benzersiz Grup", unique_combos, "teal"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tablar
        tab1, tab2, tab3 = st.tabs(["📈 Görselleştirme", "📊 Grup Dağılımları", "📋 Detay Tablo"])

        with tab1:
            if len(boyutlar) == 1:
                # 1D - Histogram
                st.markdown(f"### {boyutlar[0]} Dağılımı")
                fig = px.histogram(
                    results,
                    x=boyutlar[0],
                    color=f'{boyutlar[0]}_Grup',
                    color_discrete_sequence=CLUSTER_COLORS,
                    barmode='overlay',
                    opacity=0.7
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            elif len(boyutlar) == 2:
                # 2D - Scatter
                st.markdown(f"### {boyutlar[0]} vs {boyutlar[1]}")
                fig = px.scatter(
                    results,
                    x=boyutlar[0],
                    y=boyutlar[1],
                    color='Kombine_Grup',
                    color_discrete_sequence=CLUSTER_COLORS,
                    hover_data=[name_col] if name_col else None,
                    opacity=0.7
                )
                fig.update_traces(marker=dict(size=10))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            else:
                # 3D - Scatter
                st.markdown(f"### 3D Görünüm")
                fig = px.scatter_3d(
                    results,
                    x=boyutlar[0],
                    y=boyutlar[1],
                    z=boyutlar[2],
                    color='Kombine_Grup',
                    color_discrete_sequence=CLUSTER_COLORS,
                    hover_data=[name_col] if name_col else None,
                    opacity=0.7
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Her boyut için grup dağılımı
            for i, (boyut, n_grup) in enumerate(zip(boyutlar, grup_sayilari)):
                st.markdown(f"### {boyut} - {n_grup} Grup")

                col_chart, col_stats = st.columns([2, 1])

                with col_chart:
                    # Box plot
                    fig = px.box(
                        results,
                        x=f'{boyut}_Grup',
                        y=boyut,
                        color=f'{boyut}_Grup',
                        color_discrete_sequence=CLUSTER_COLORS
                    )
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    # Grup istatistikleri
                    stats = results.groupby(f'{boyut}_Grup')[boyut].agg(['count', 'mean', 'min', 'max']).round(2)
                    stats.columns = ['Adet', 'Ortalama', 'Min', 'Max']
                    st.dataframe(stats, use_container_width=True)

                st.markdown("---")

            # Kombine grup dağılımı
            st.markdown("### Kombine Grup Dağılımı")
            combo_counts = results['Kombine_Grup'].value_counts().reset_index()
            combo_counts.columns = ['Grup', 'Adet']

            fig = px.bar(
                combo_counts.head(20),
                x='Grup',
                y='Adet',
                color='Adet',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("### Detay Listesi")

            # Filtreler
            filter_cols = st.columns(len(boyutlar) + 1)

            filtered = results.copy()

            for i, boyut in enumerate(boyutlar):
                with filter_cols[i]:
                    grup_col = f'{boyut}_Grup'
                    selected = st.multiselect(
                        f"{boyut} Grup",
                        options=sorted(results[grup_col].unique()),
                        default=sorted(results[grup_col].unique())
                    )
                    filtered = filtered[filtered[grup_col].isin(selected)]

            with filter_cols[-1]:
                if name_col:
                    search = st.text_input("Ara", placeholder="İsim ara...")
                    if search:
                        filtered = filtered[filtered[name_col].astype(str).str.contains(search, case=False, na=False)]

            # Tablo göster
            # Sütun sırası: name_col, boyutlar, grup kolonları, kombine grup
            display_cols = [name_col] if name_col else []
            display_cols += boyutlar
            display_cols += [f'{b}_Grup' for b in boyutlar]
            display_cols.append('Kombine_Grup')

            # Diğer kolonları da ekle
            other_cols = [c for c in results.columns if c not in display_cols]
            display_cols += other_cols

            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                height=500
            )

            st.markdown(f"**Gösterilen:** {len(filtered)} / {len(results)} kayıt")

            # Export
            st.markdown("---")
            st.markdown("### 📥 İndir")

            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                # Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results[display_cols].to_excel(writer, index=False, sheet_name='Gruplu Veri')

                    # Her boyut için özet
                    for boyut in boyutlar:
                        stats = results.groupby(f'{boyut}_Grup')[boyut].agg(['count', 'mean', 'min', 'max']).round(2)
                        stats.to_excel(writer, sheet_name=f'{boyut[:20]} Özet')

                st.download_button(
                    label="📥 Excel İndir",
                    data=buffer.getvalue(),
                    file_name=f"gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col_exp2:
                # CSV
                csv = results[display_cols].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 CSV İndir",
                    data=csv,
                    file_name=f"gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    else:
        st.info("👆 Sol panelden ayarları yapıp 'Grupla' butonuna tıklayın")


if __name__ == "__main__":
    main()
