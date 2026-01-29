"""
Mağaza Gruplama Analizi - K-Means Clustering
Thorius Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
from datetime import datetime

# Sayfa ayarları
st.set_page_config(
    page_title="Mağaza Gruplama Analizi",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Thorius teması
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a2c7a 50%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(74, 44, 122, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-card.blue {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    }

    .metric-card.purple {
        background: linear-gradient(135deg, #4a2c7a 0%, #7b4397 100%);
    }

    .metric-card.teal {
        background: linear-gradient(135deg, #0d7377 0%, #14919b 100%);
    }

    .metric-card.indigo {
        background: linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .cluster-card {
        background: linear-gradient(145deg, #f8f9ff 0%, #e8ecff 100%);
        border-left: 5px solid;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(90deg, #1e3a5f 0%, #4a2c7a 100%);
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: rgba(255,255,255,0.7);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e3a5f !important;
    }

    .sidebar .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f7 100%);
        border-left: 4px solid #1e3a5f;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8ecff 100%);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Cluster renkleri
CLUSTER_COLORS = [
    '#667eea', '#764ba2', '#0d7377', '#f093fb', '#f5576c',
    '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140'
]

def create_sample_data():
    """Örnek veri oluştur"""
    np.random.seed(42)
    n_stores = 150

    stores = [f"Mağaza_{i+1}" for i in range(n_stores)]
    cities = np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Konya'], n_stores)

    data = {
        'Mağaza_Adı': stores,
        'Şehir': cities,
        'Kapasite_m2': np.random.randint(50, 500, n_stores),
        'Stok_Devir_Hızı': np.round(np.random.uniform(2, 12, n_stores), 2),
        'Günlük_Satış_Adet': np.random.randint(20, 500, n_stores),
        'Ortalama_Sepet_TL': np.round(np.random.uniform(150, 800, n_stores), 2),
        'Personel_Sayısı': np.random.randint(3, 25, n_stores),
        'Aylık_Ciro_TL': np.random.randint(50000, 2000000, n_stores),
        'Müşteri_Sayısı': np.random.randint(500, 10000, n_stores),
        'Karlılık_Oranı': np.round(np.random.uniform(5, 35, n_stores), 2),
        'Doluluk_Oranı': np.round(np.random.uniform(40, 95, n_stores), 2),
        'Metrekare_Verimlilik': np.round(np.random.uniform(100, 1000, n_stores), 2)
    }

    return pd.DataFrame(data)

def load_data(uploaded_file):
    """Dosya yükle ve oku"""
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

def perform_clustering(df, selected_metrics, n_clusters, normalize=True):
    """K-Means clustering uygula"""
    X = df[selected_metrics].copy()
    X = X.fillna(X.mean())

    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, clusters)
    else:
        silhouette = 0

    return clusters, kmeans, silhouette, X_scaled

def create_3d_scatter(df, metrics, clusters):
    """3D Scatter plot oluştur"""
    if len(metrics) < 3:
        metrics = metrics + [metrics[0]] * (3 - len(metrics))

    fig = go.Figure(data=[go.Scatter3d(
        x=df[metrics[0]],
        y=df[metrics[1]],
        z=df[metrics[2]],
        mode='markers',
        marker=dict(
            size=8,
            color=clusters,
            colorscale=[[i/(len(CLUSTER_COLORS)-1), c] for i, c in enumerate(CLUSTER_COLORS[:max(clusters)+1])],
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=[f"Cluster {c}" for c in clusters],
        hovertemplate=
            f"<b>{metrics[0]}:</b> %{{x:.2f}}<br>" +
            f"<b>{metrics[1]}:</b> %{{y:.2f}}<br>" +
            f"<b>{metrics[2]}:</b> %{{z:.2f}}<br>" +
            "<b>Cluster:</b> %{text}<extra></extra>"
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title=metrics[0],
            yaxis_title=metrics[1],
            zaxis_title=metrics[2],
            bgcolor='rgba(248,249,255,0.95)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_scatter_matrix(df, metrics, clusters):
    """Scatter matrix oluştur"""
    df_plot = df[metrics].copy()
    df_plot['Cluster'] = [f"Cluster {c}" for c in clusters]

    fig = px.scatter_matrix(
        df_plot,
        dimensions=metrics[:5],  # Max 5 metrik
        color='Cluster',
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.7
    )

    fig.update_layout(
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,255,0.95)'
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)

    return fig

def create_box_plots(df, metrics, clusters):
    """Cluster bazında box plotlar"""
    df_plot = df[metrics].copy()
    df_plot['Cluster'] = [f"Cluster {c}" for c in clusters]

    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=metrics)

    for i, metric in enumerate(metrics):
        row = i // cols + 1
        col = i % cols + 1

        for cluster in sorted(df_plot['Cluster'].unique()):
            cluster_data = df_plot[df_plot['Cluster'] == cluster][metric]
            cluster_num = int(cluster.split()[-1])

            fig.add_trace(
                go.Box(
                    y=cluster_data,
                    name=cluster,
                    marker_color=CLUSTER_COLORS[cluster_num % len(CLUSTER_COLORS)],
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=300 * rows,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,255,0.95)',
        showlegend=True
    )

    return fig

def create_radar_chart(cluster_profiles, metrics):
    """Radar chart oluştur"""
    fig = go.Figure()

    # Değerleri normalize et (0-1 arasına)
    profile_df = cluster_profiles[metrics]
    normalized = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min() + 0.0001)

    for i, (idx, row) in enumerate(normalized.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f'Cluster {idx}',
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor='rgba(248,249,255,0.95)'
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def render_metric_card(label, value, color_class=""):
    """Metrik kartı HTML"""
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def render_cluster_summary_card(cluster_id, count, avg_values, color):
    """Cluster özet kartı"""
    metrics_html = ""
    for metric, value in list(avg_values.items())[:4]:
        short_metric = metric[:20] + "..." if len(metric) > 20 else metric
        metrics_html += f"<div><strong>{short_metric}:</strong> {value:,.2f}</div>"

    return f"""
    <div class="cluster-card" style="border-left-color: {color};">
        <h3 style="color: {color}; margin-top: 0;">Cluster {cluster_id}</h3>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;"><strong>{count}</strong> Mağaza</p>
        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 1rem 0;">
        {metrics_html}
    </div>
    """

# Ana uygulama
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏪 Mağaza Gruplama Analizi</h1>
        <p>K-Means Clustering ile Akıllı Segmentasyon | Thorius Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Veri Kaynağı")

        uploaded_file = st.file_uploader(
            "Excel/CSV Dosya Yükle",
            type=['xlsx', 'xls', 'csv'],
            help="Mağaza verilerinizi içeren dosyayı yükleyin"
        )

        use_sample = st.checkbox("Örnek veri kullan", value=False)

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success(f"✅ {len(df)} satır yüklendi")
        elif use_sample:
            df = create_sample_data()
            st.info("📊 Örnek veri kullanılıyor")
        else:
            df = None

        if df is not None:
            st.markdown("---")
            st.markdown("### 📊 Analiz Ayarları")

            numeric_cols = get_numeric_columns(df)

            # İsim sütununu bul
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            name_col = text_cols[0] if text_cols else None

            selected_metrics = st.multiselect(
                "Metrikler",
                options=numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                help="Gruplama için kullanılacak metrikleri seçin"
            )

            n_clusters = st.slider(
                "Cluster Sayısı",
                min_value=2,
                max_value=10,
                value=4,
                help="Oluşturulacak grup sayısı"
            )

            normalize = st.radio(
                "Normalizasyon",
                options=["Evet", "Hayır"],
                horizontal=True,
                help="Verileri standartlaştır"
            ) == "Evet"

            st.markdown("---")

            run_clustering = st.button("🚀 Gruplama Yap", use_container_width=True)
        else:
            selected_metrics = []
            run_clustering = False
            name_col = None

    # Ana içerik
    if df is None:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Hoş Geldiniz!</h3>
            <p>Mağaza gruplama analizine başlamak için:</p>
            <ol>
                <li>Sol panelden Excel/CSV dosyanızı yükleyin</li>
                <li>Veya "Örnek veri kullan" seçeneğini işaretleyin</li>
                <li>Analiz metriklerini seçin</li>
                <li>"Gruplama Yap" butonuna tıklayın</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    if len(selected_metrics) < 2:
        st.warning("⚠️ Lütfen en az 2 metrik seçin")
        return

    # Session state
    if 'clusters' not in st.session_state:
        st.session_state.clusters = None
        st.session_state.silhouette = None
        st.session_state.kmeans = None

    if run_clustering:
        with st.spinner("Clustering yapılıyor..."):
            clusters, kmeans, silhouette, X_scaled = perform_clustering(
                df, selected_metrics, n_clusters, normalize
            )
            st.session_state.clusters = clusters
            st.session_state.silhouette = silhouette
            st.session_state.kmeans = kmeans
            st.session_state.X_scaled = X_scaled

    if st.session_state.clusters is not None:
        clusters = st.session_state.clusters
        silhouette = st.session_state.silhouette

        # Sonuç dataframe
        df_result = df.copy()
        df_result['Cluster'] = clusters

        # Üst metrikler
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(render_metric_card("Toplam Mağaza", len(df), "blue"), unsafe_allow_html=True)

        with col2:
            st.markdown(render_metric_card("Cluster Sayısı", n_clusters, "purple"), unsafe_allow_html=True)

        with col3:
            st.markdown(render_metric_card("Silhouette Score", f"{silhouette:.3f}", "teal"), unsafe_allow_html=True)

        with col4:
            cluster_counts = pd.Series(clusters).value_counts()
            biggest_cluster = cluster_counts.idxmax()
            st.markdown(render_metric_card("En Büyük Cluster", f"#{biggest_cluster} ({cluster_counts[biggest_cluster]})", "indigo"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tablar
        tab1, tab2, tab3 = st.tabs(["📈 Görselleştirme", "🎯 Cluster Profilleri", "📋 Detay Tablo"])

        with tab1:
            st.markdown("### 3D Cluster Dağılımı")
            fig_3d = create_3d_scatter(df, selected_metrics, clusters)
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("### 2D Scatter Matrix")
            fig_matrix = create_scatter_matrix(df, selected_metrics, clusters)
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.markdown("### Cluster Bazında Dağılımlar")
            fig_box = create_box_plots(df, selected_metrics[:6], clusters)  # Max 6 metrik
            st.plotly_chart(fig_box, use_container_width=True)

        with tab2:
            # Cluster profilleri hesapla
            cluster_profiles = df_result.groupby('Cluster')[selected_metrics].mean()
            cluster_counts = df_result['Cluster'].value_counts().sort_index()

            st.markdown("### Cluster Özet Kartları")

            cols = st.columns(min(4, n_clusters))
            for i, (cluster_id, row) in enumerate(cluster_profiles.iterrows()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    avg_values = row.to_dict()
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    st.markdown(
                        render_cluster_summary_card(cluster_id, cluster_counts[cluster_id], avg_values, color),
                        unsafe_allow_html=True
                    )

            st.markdown("---")

            col_radar, col_table = st.columns([1, 1])

            with col_radar:
                st.markdown("### Cluster Profil Karşılaştırması")
                display_metrics = selected_metrics[:8]  # Radar için max 8 metrik
                fig_radar = create_radar_chart(cluster_profiles, display_metrics)
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_table:
                st.markdown("### Cluster Merkezleri")

                # Merkez tablosu
                centers_df = cluster_profiles.round(2)
                centers_df.insert(0, 'Mağaza Sayısı', cluster_counts)

                st.dataframe(
                    centers_df.style.background_gradient(cmap='Blues', subset=selected_metrics),
                    use_container_width=True,
                    height=400
                )

        with tab3:
            st.markdown("### Mağaza Detay Listesi")

            # Filtreler
            col_filter1, col_filter2 = st.columns([1, 3])

            with col_filter1:
                selected_clusters = st.multiselect(
                    "Cluster Filtre",
                    options=sorted(df_result['Cluster'].unique()),
                    default=sorted(df_result['Cluster'].unique())
                )

            with col_filter2:
                if name_col:
                    search_text = st.text_input("Mağaza Ara", placeholder="Mağaza adı ile ara...")
                else:
                    search_text = ""

            # Filtreleme
            filtered_df = df_result[df_result['Cluster'].isin(selected_clusters)]

            if search_text and name_col:
                filtered_df = filtered_df[filtered_df[name_col].str.contains(search_text, case=False, na=False)]

            # Tablo
            st.dataframe(
                filtered_df.style.applymap(
                    lambda x: f'background-color: {CLUSTER_COLORS[x % len(CLUSTER_COLORS)]}22' if isinstance(x, (int, np.integer)) and 'Cluster' in str(x) else '',
                    subset=['Cluster']
                ),
                use_container_width=True,
                height=500
            )

            st.markdown(f"**Gösterilen:** {len(filtered_df)} / {len(df_result)} mağaza")

            # Export
            st.markdown("---")
            col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])

            with col_exp1:
                # Excel export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Mağaza Grupları')
                    cluster_profiles.to_excel(writer, sheet_name='Cluster Profilleri')

                st.download_button(
                    label="📥 Excel İndir",
                    data=buffer.getvalue(),
                    file_name=f"magaza_gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col_exp2:
                # CSV export
                csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 CSV İndir",
                    data=csv,
                    file_name=f"magaza_gruplama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("👆 Sol panelden ayarları yapıp 'Gruplama Yap' butonuna tıklayın")

if __name__ == "__main__":
    main()
