"""
Working Capital Simulator - Streamlit app
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.stats.mstats import winsorize


# ============================================================
# PAGE CONFIG + CUSTOM STYLE
# ============================================================
st.set_page_config(
    page_title="WC Simulator",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1300px;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    h2, h3 {
        color: #1e3a8a;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #1e3a8a;
        font-size: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a8a !important;
        color: white !important;
    }
    .receta-box {
        background: #f8fafc;
        border-left: 4px solid #1e3a8a;
        padding: 12px 18px;
        border-radius: 4px;
        margin: 8px 0;
        font-family: monospace;
    }
    .alerta-estructural {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 12px 18px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PIPELINE: cargar datos + clustering + targets
# ============================================================
@st.cache_data
def cargar_y_clusterizar():
    df = pd.read_csv("dataset_finanzas_real.csv")
    features = ["cash_pct", "receivables_pct", "inventory_pct", "dso", "dio", "dpo", "ccc"]

    X_wins = df[features].copy()
    for col in features:
        X_wins[col] = winsorize(X_wins[col], limits=[0.025, 0.025])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_wins)

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=20)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    targets = {}
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        n_top = max(3, int(len(sub) * 0.30))
        top = sub.nsmallest(n_top, "ccc")
        targets[c] = {
            "dio_target": float(top["dio"].median()),
            "dso_target": float(top["dso"].median()),
            "dpo_target": float(top["dpo"].median()),
            "ccc_target": float(top["ccc"].median()),
            "n_train": n_top,
            "empresas_top": sorted(set(top["ticker"])),
        }

    cluster_profile = df.groupby("cluster")[features].mean()
    cluster_eficiente = int(cluster_profile["ccc"].idxmin())
    ccc_eficiente_global = float(cluster_profile.loc[cluster_eficiente, "ccc"])

    return df, targets, cluster_eficiente, ccc_eficiente_global, cluster_profile


df, targets_por_cluster, cluster_eficiente, ccc_eficiente_global, cluster_profile = cargar_y_clusterizar()


# ============================================================
# SIDEBAR: selector de empresa y año
# ============================================================
with st.sidebar:
    st.title("Configuración")
    st.markdown("---")

    tickers = sorted(df["ticker"].unique())
    default_idx = tickers.index("WMT") if "WMT" in tickers else 0
    ticker = st.selectbox("Empresa", tickers, index=default_idx)

    df_emp = df[df["ticker"] == ticker]
    años_disponibles = sorted(df_emp["fiscal_year"].unique(), reverse=True)
    año = st.selectbox("Año fiscal", años_disponibles, index=0)

    emp = df_emp[df_emp["fiscal_year"] == año].iloc[0]
    c = int(emp["cluster"])
    t = targets_por_cluster[c]

    st.markdown("---")
    st.markdown("**Cluster asignado**")
    st.info(f"Cluster {c}")
    st.caption(f"Top-30% del cluster: {', '.join(t['empresas_top'])}")

    st.markdown("---")
    st.markdown("**Acerca**")
    st.caption(
        "Simulador de gestión de capital de trabajo. "
        "Targets calculados como mediana del top-30% por menor CCC dentro del cluster."
    )


# ============================================================
# HEADER
# ============================================================
st.title("Working Capital Simulator")
st.markdown(
    f"<h3 style='color:#64748b; font-weight:400; margin-top:-10px;'>"
    f"{ticker} &nbsp;·&nbsp; {emp['sector']} &nbsp;·&nbsp; FY {año}"
    f"</h3>",
    unsafe_allow_html=True
)


# ============================================================
# FILA DE KPIs
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Revenue", f"${emp['revenue']/1e9:.2f}B")
col2.metric("COGS", f"${emp['cogs']/1e9:.2f}B")
col3.metric("CCC actual", f"{emp['ccc']:.1f}d",
            delta=f"{emp['ccc'] - t['ccc_target']:+.1f}d vs target",
            delta_color="inverse")
col4.metric("CCC target cluster", f"{t['ccc_target']:.1f}d")
col5.metric("CCC cluster eficiente", f"{ccc_eficiente_global:.1f}d")

st.markdown("---")


# ============================================================
# SLIDERS
# ============================================================
st.subheader("Simulación — moveé las palancas")

col_dio, col_dso, col_dpo = st.columns(3)
with col_dio:
    DIO = st.slider("DIO (Days Inventory Outstanding)",
                    min_value=0.0, max_value=500.0,
                    value=float(emp["dio"]), step=1.0,
                    help="Días para vender el inventario. Bajar libera cash del depósito.")
with col_dso:
    DSO = st.slider("DSO (Days Sales Outstanding)",
                    min_value=0.0, max_value=300.0,
                    value=float(emp["dso"]), step=1.0,
                    help="Días para cobrar al cliente. Bajar libera cash del limbo de cobros.")
with col_dpo:
    DPO = st.slider("DPO (Days Payable Outstanding)",
                    min_value=0.0, max_value=300.0,
                    value=float(emp["dpo"]), step=1.0,
                    help="Días para pagar a proveedores. SUBIR libera cash (los proveedores te financian).")

ccc_n = DIO + DSO - DPO
cash_liberado = (emp["ccc"] - ccc_n) * (emp["revenue"] / 365)


# ============================================================
# RESULTADO PRINCIPAL: cash liberado
# ============================================================
col_resultado, col_gauge = st.columns([1, 1.5])

with col_resultado:
    st.subheader("Resultado")
    if cash_liberado >= 0:
        st.metric(
            "Cash liberado",
            f"+${cash_liberado/1e6:,.0f}M USD",
            delta=f"ΔCCC = {ccc_n - emp['ccc']:+.1f} días",
            delta_color="inverse"
        )
    else:
        st.metric(
            "Cash atrapado adicional",
            f"-${abs(cash_liberado)/1e6:,.0f}M USD",
            delta=f"ΔCCC = {ccc_n - emp['ccc']:+.1f} días",
            delta_color="inverse"
        )

    st.caption(
        f"Cash liberado = (CCC actual − CCC nuevo) × (Revenue/365) "
        f"= ({emp['ccc']:.1f} − {ccc_n:.1f}) × ${emp['revenue']/365/1e6:,.1f}M"
    )

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ccc_n,
        delta={"reference": emp["ccc"], "decreasing": {"color": "#059669"}, "increasing": {"color": "#dc2626"}},
        title={"text": "CCC simulado (días)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [min(-100, ccc_n - 20), max(200, ccc_n + 20)], "tickwidth": 1},
            "bar": {"color": "#1e3a8a", "thickness": 0.7},
            "steps": [
                {"range": [-100, ccc_eficiente_global], "color": "#bbf7d0"},
                {"range": [ccc_eficiente_global, t["ccc_target"]], "color": "#fef3c7"},
                {"range": [t["ccc_target"], 250], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#1e3a8a", "width": 3},
                "thickness": 0.85,
                "value": t["ccc_target"],
            },
        },
    ))
    fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=70, b=30))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ============================================================
# TABS: tabla ratios | tabla USD | receta | analisis cluster
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Ratios (días)", "Niveles en USD", "Receta", "Cluster"])

# --- TAB 1: ratios en días ---
with tab1:
    tabla_ratios = pd.DataFrame({
        "Métrica": ["DIO", "DSO", "DPO", "CCC"],
        "Actual": [emp["dio"], emp["dso"], emp["dpo"], emp["ccc"]],
        "Simulado": [DIO, DSO, DPO, ccc_n],
        "Δ": [DIO - emp["dio"], DSO - emp["dso"], DPO - emp["dpo"], ccc_n - emp["ccc"]],
        "Target cluster": [t["dio_target"], t["dso_target"], t["dpo_target"], t["ccc_target"]],
    }).round(1)

    # Coloreado direccional: DPO invertido (subir = bueno = verde)
    def color_delta_ratios(row):
        styles = [""] * len(row)
        val = row["Δ"]
        metric = row["Métrica"]
        delta_idx = list(row.index).index("Δ")
        if metric == "DPO":
            # subir DPO es BUENO
            if val > 0.5:
                styles[delta_idx] = "background-color: #bbf7d0; color: #064e3b; font-weight: 600;"
            elif val < -0.5:
                styles[delta_idx] = "background-color: #fecaca; color: #7f1d1d; font-weight: 600;"
        else:
            # DIO, DSO, CCC: bajar es BUENO
            if val < -0.5:
                styles[delta_idx] = "background-color: #bbf7d0; color: #064e3b; font-weight: 600;"
            elif val > 0.5:
                styles[delta_idx] = "background-color: #fecaca; color: #7f1d1d; font-weight: 600;"
        return styles

    st.dataframe(
        tabla_ratios.style
            .format({"Actual": "{:.1f}", "Simulado": "{:.1f}", "Δ": "{:+.1f}", "Target cluster": "{:.1f}"})
            .apply(color_delta_ratios, axis=1)
            .set_properties(**{"font-size": "1rem"}),
        hide_index=True,
        use_container_width=True,
    )

# --- TAB 2: niveles USD ---
with tab2:
    cogs = float(emp["cogs"])
    rev = float(emp["revenue"])
    inv_opt = t["dio_target"] * cogs / 365
    ar_opt = t["dso_target"] * rev / 365
    ap_opt = t["dpo_target"] * cogs / 365

    tabla_usd = pd.DataFrame({
        "Componente": ["Inventario", "Cuentas por cobrar", "Cuentas por pagar"],
        "Actual ($M)": [emp["inventory"] / 1e6, emp["accounts_receivable"] / 1e6, emp["accounts_payable"] / 1e6],
        "Óptimo ($M)": [inv_opt / 1e6, ar_opt / 1e6, ap_opt / 1e6],
        "Brecha ($M)": [
            (emp["inventory"] - inv_opt) / 1e6,
            (emp["accounts_receivable"] - ar_opt) / 1e6,
            (emp["accounts_payable"] - ap_opt) / 1e6,
        ],
    }).round(0)

    # Coloreado direccional: AP invertido (más AP = bueno, te financia el proveedor)
    def color_brecha_usd(row):
        styles = [""] * len(row)
        val = row["Brecha ($M)"]
        componente = row["Componente"]
        brecha_idx = list(row.index).index("Brecha ($M)")
        umbral = 1  # M USD
        if componente == "Cuentas por pagar":
            # AP: brecha positiva = financiamiento mayor al benchmark = BUENO
            if val > umbral:
                styles[brecha_idx] = "background-color: #bbf7d0; color: #064e3b; font-weight: 600;"
            elif val < -umbral:
                styles[brecha_idx] = "background-color: #fecaca; color: #7f1d1d; font-weight: 600;"
        else:
            # Inventario y AR: brecha positiva = exceso atrapado = MALO
            if val > umbral:
                styles[brecha_idx] = "background-color: #fecaca; color: #7f1d1d; font-weight: 600;"
            elif val < -umbral:
                styles[brecha_idx] = "background-color: #bbf7d0; color: #064e3b; font-weight: 600;"
        return styles

    st.dataframe(
        tabla_usd.style
            .format({"Actual ($M)": "${:,.0f}M", "Óptimo ($M)": "${:,.0f}M", "Brecha ($M)": "${:+,.0f}M"})
            .apply(color_brecha_usd, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Inventario y AR: brecha positiva = exceso de capital atrapado (malo). "
        "AP: brecha positiva = el proveedor te financia más que el benchmark (bueno)."
    )

    # Bar chart comparativo
    df_bar = tabla_usd.melt(id_vars="Componente",
                            value_vars=["Actual ($M)", "Óptimo ($M)"],
                            var_name="Tipo", value_name="USD ($M)")
    fig_bar = px.bar(df_bar, x="Componente", y="USD ($M)", color="Tipo",
                     barmode="group",
                     color_discrete_map={"Actual ($M)": "#3b82f6", "Óptimo ($M)": "#10b981"},
                     height=320)
    fig_bar.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: receta operativa ---
with tab3:
    st.markdown(f"**Receta para llegar al target del cluster {c}:**")

    def receta_palanca(nombre, actual, target, mejor_baja, usd_por_dia):
        if mejor_baja:
            diff = actual - target
            verbo = "Bajar"
        else:
            diff = target - actual
            verbo = "Subir"

        if diff > 1:
            usd_efecto = diff * usd_por_dia / 1e6
            return f"<div class='receta-box'>{verbo} <b>{nombre}</b> en <b>{diff:.1f}</b> días &nbsp;→&nbsp; objetivo <b>{target:.1f}d</b> &nbsp;·&nbsp; libera ~<b>${usd_efecto:,.0f}M</b> USD</div>"
        elif diff < -1:
            return f"<div class='receta-box' style='border-color:#10b981'>{nombre}: tu nivel ({actual:.1f}d) <b>ya es mejor</b> que el target ({target:.1f}d). Mantener.</div>"
        else:
            return f"<div class='receta-box' style='border-color:#94a3b8'>{nombre}: <b>alineado</b> con el target ({actual:.1f}d vs {target:.1f}d). Sin acción.</div>"

    st.markdown(receta_palanca("DIO", emp["dio"], t["dio_target"], True, cogs / 365), unsafe_allow_html=True)
    st.markdown(receta_palanca("DSO", emp["dso"], t["dso_target"], True, rev / 365), unsafe_allow_html=True)
    st.markdown(receta_palanca("DPO", emp["dpo"], t["dpo_target"], False, cogs / 365), unsafe_allow_html=True)

    # Brecha estructural
    brecha_estructural = t["ccc_target"] - ccc_eficiente_global
    if brecha_estructural > 10:
        st.markdown(
            f"<div class='alerta-estructural'>"
            f"<b>Brecha estructural detectada.</b><br>"
            f"Aún llegando al target operativo del cluster (CCC={t['ccc_target']:.0f}d), "
            f"quedan <b>{brecha_estructural:.0f} días</b> de brecha vs el cluster más eficiente "
            f"(CCC={ccc_eficiente_global:.0f}d).<br>"
            f"Esa brecha es estructural: requiere cambiar modelo de negocio, no solo palancas operativas."
            f"</div>",
            unsafe_allow_html=True,
        )

# --- TAB 4: contexto del cluster ---
with tab4:
    st.markdown(f"**Composición del cluster {c}**")

    df_cluster = df[df["cluster"] == c]
    empresas_cluster = sorted(set(df_cluster["ticker"]))
    sectores_cluster = df_cluster["sector"].value_counts()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Empresas presentes**")
        st.write(", ".join(empresas_cluster))
        st.caption(f"{len(empresas_cluster)} empresas únicas, {len(df_cluster)} observaciones empresa-año")

    with col_b:
        st.markdown("**Distribución por sector**")
        st.dataframe(sectores_cluster.rename("Observaciones").to_frame(), use_container_width=True)

    st.markdown("---")
    st.markdown("**Centroide del cluster (promedio de cada feature)**")
    features = ["cash_pct", "receivables_pct", "inventory_pct", "dso", "dio", "dpo", "ccc"]
    centroide = df_cluster[features].mean().round(2).to_frame("Valor promedio")
    st.dataframe(centroide, use_container_width=True)

    st.markdown("---")
    st.markdown("**Empresas usadas para calcular el target (top-30% por menor CCC)**")
    st.success(", ".join(t["empresas_top"]))
    st.caption(f"n = {t['n_train']} observaciones empresa-año")
