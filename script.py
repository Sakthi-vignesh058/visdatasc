import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Climate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center;}
    h2, h3 {color: #34495e;}
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("global_warming_dataset.csv")

df = load_data()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🎛️ Filters")

countries = st.sidebar.multiselect(
    "Select Countries",
    options=df["Country"].unique(),
    default=df["Country"].unique()[:5]
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

filtered_df = df[
    (df["Country"].isin(countries)) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1])
]

# -------------------- TITLE --------------------
st.title("🌍 Climate Change Analysis Dashboard")
st.markdown("---")

# ==================== MODELING SECTION ====================
st.header("📊 Modeling Results")

col_model1, col_model2 = st.columns(2)

# ----------- CLUSTERING -----------
with col_model1:
    st.subheader("🔵 Clustering Analysis (K-Means)")

    features = [
        "Temperature_Anomaly",
        "CO2_Emissions",
        "Renewable_Energy_Usage",
        "Per_Capita_Emissions"
    ]

    X = filtered_df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = st.slider("Number of Clusters", 2, 5, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters,
        "Country": filtered_df.loc[X.index, "Country"].values,
        "Year": filtered_df.loc[X.index, "Year"].values
    })

    fig_cluster = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Country", "Year"],
        title="Climate Clusters (PCA Projection)"
    )

    fig_cluster.update_layout(height=400)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.info(
        f"Variance Explained — PC1: {pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2: {pca.explained_variance_ratio_[1]:.1%}"
    )

# ----------- REGRESSION -----------
with col_model2:
    st.subheader("📈 Regression Analysis")

    reg_df = filtered_df[["CO2_Emissions", "Temperature_Anomaly"]].dropna()

    X_reg = reg_df[["CO2_Emissions"]]
    y_reg = reg_df["Temperature_Anomaly"]

    model = LinearRegression()
    model.fit(X_reg, y_reg)

    y_pred = model.predict(X_reg)
    r2 = model.score(X_reg, y_reg)

    fig_reg = go.Figure()

    fig_reg.add_trace(go.Scatter(
        x=reg_df["CO2_Emissions"] / 1e9,
        y=y_reg,
        mode="markers",
        name="Data"
    ))

    fig_reg.add_trace(go.Scatter(
        x=reg_df["CO2_Emissions"] / 1e9,
        y=y_pred,
        mode="lines",
        name="Regression"
    ))

    fig_reg.update_layout(
        title=f"CO₂ vs Temperature (R² = {r2:.3f})",
        xaxis_title="CO₂ Emissions (Billion tons)",
        yaxis_title="Temperature Anomaly (°C)",
        height=400
    )

    st.plotly_chart(fig_reg, use_container_width=True)

# ==================== DASHBOARD ====================
st.markdown("---")
st.header("📱 Interactive Dashboard with Brushing & Linking")

# -------------------- METRICS --------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Temp Anomaly", f"{filtered_df['Temperature_Anomaly'].mean():.2f} °C")
col2.metric("Total CO₂", f"{filtered_df['CO2_Emissions'].sum()/1e9:.1f} B tons")
col3.metric("Avg Renewable", f"{filtered_df['Renewable_Energy_Usage'].mean():.1f} %")
col4.metric("Extreme Events", int(filtered_df["Extreme_Weather_Events"].sum()))

# ==================== BRUSHING & LINKING ====================
st.subheader("🌊 Sea Level Rise vs Forest Area (Brush to Link)")

brush_fig = px.scatter(
    filtered_df,
    x="Forest_Area",
    y="Sea_Level_Rise",
    color="Country",
    size="Population",
    hover_data=["Year", "Temperature_Anomaly"],
    opacity=0.6
)

brush_fig.update_layout(
    dragmode="lasso",
    height=400
)

brush_event = st.plotly_chart(
    brush_fig,
    use_container_width=True,
    key="brush_plot"
)

# -------------------- CAPTURE BRUSH --------------------
selected_points = st.session_state.get("brush_plot", {}).get("selected_points")

if selected_points:
    indices = [p["pointIndex"] for p in selected_points]
    brushed_df = filtered_df.iloc[indices]
else:
    brushed_df = filtered_df

# ==================== LINKED LINE CHART ====================
st.subheader("🌡️ Temperature Trend (Linked View)")

linked_trend = brushed_df.groupby("Year")["Temperature_Anomaly"].mean().reset_index()

fig_linked = px.line(
    linked_trend,
    x="Year",
    y="Temperature_Anomaly",
    markers=True
)

fig_linked.update_layout(height=350)
st.plotly_chart(fig_linked, use_container_width=True)

# ==================== LINKED TABLE ====================
st.subheader("📋 Brushed Data Records")

st.dataframe(
    brushed_df[
        ["Country", "Year", "Forest_Area", "Sea_Level_Rise", "Temperature_Anomaly"]
    ],
    use_container_width=True
)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "**Data Source:** Climate Change Analysis Dataset | "
    "**Dashboard:** Streamlit & Plotly | "
    "**Interaction:** Brushing & Linking Enabled"
)
