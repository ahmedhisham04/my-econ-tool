import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import add_constant

# ─────────────────────────────────────────────────────────────────
# NEXUS DESIGN SYSTEM (Professional, Neutral, High-End)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nexus Macro-Financial Suite", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Roboto+Mono:wght@400;500&display=swap');

    :root {
        --nexus-bg: #0F172A;
        --nexus-card: #1E293B;
        --nexus-accent: #3B82F6;
        --nexus-border: #334155;
        --nexus-text: #F8FAFC;
    }

    .stApp { background-color: var(--nexus-bg); color: var(--nexus-text); font-family: 'Inter', sans-serif; }

    /* Professional Header */
    .header-bar {
        background-color: var(--nexus-card);
        padding: 1.5rem 2rem;
        border-bottom: 1px solid var(--nexus-border);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-name { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: var(--nexus-text); }
    .status-badge { font-family: 'Roboto Mono', monospace; font-size: 11px; padding: 4px 12px; background: #064E3B; color: #34D399; border-radius: 2px; }

    /* The Setup Wizard Box */
    .setup-box {
        background: var(--nexus-card);
        border: 1px solid var(--nexus-border);
        padding: 2rem;
        border-radius: 4px;
        margin-bottom: 2rem;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 30px; }
    .stTabs [data-baseweb="tab"] { font-family: 'Roboto Mono', monospace; font-size: 13px; color: #94A3B8 !important; }
    .stTabs [aria-selected="true"] { color: var(--nexus-accent) !important; border-bottom: 2px solid var(--nexus-accent) !important; }

    /* Tables & Frames */
    .stDataFrame { border: 1px solid var(--nexus-border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="brand-name">NEXUS ECONOMETRIC SUITE <span style='font-weight:300; opacity:0.6'>v2.4</span></div>
    <div class="status-badge">RESEARCH KERNEL ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# INITIALIZATION & CLEANING
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DATA INGESTION")
    uploaded = st.file_uploader("Upload Raw Data (CSV/XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("Confidence Level", [0.01, 0.05, 0.10], index=1)

if uploaded:
    if uploaded.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)

    # STEP 1: KILL THE GHOST ROWS IMMEDIATELY
    df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')

    # STEP 2: THE CONFIGURATION WIZARD
    st.markdown("### SYSTEM INITIALIZATION: COLUMN MAPPING")
    
    with st.container():
        st.markdown("<div class='setup-box'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            time_col = st.selectbox("Identify Temporal Index (Year)", df_raw.columns)
        with c2:
            freq = st.selectbox("Analysis Frequency", ["Annual", "Quarterly", "Monthly"])
        
        # User explicitly chooses which numeric variables to include
        potential_vars = [c for c in df_raw.columns if c != time_col]
        selected_vars = st.multiselect("Select Variables for Analysis", potential_vars, default=potential_vars[:3])
        
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 3: STRICT DATA ENFORCEMENT
    # This prevents the "3,000 observations" error
    df = df_raw.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]) # Deletes any row where the Year isn't a number
    df = df.sort_values(by=time_col)
    
    # Final cleanup of selected variables
    for v in selected_vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    df = df.dropna(subset=selected_vars).reset_index(drop=True)
    df = df.set_index(time_col)

    # ─────────────────────────────────────────────────────────────────
    # ANALYTICS WORKBENCH
    # ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["[ DATA PREVIEW ]", "[ UNIT ROOT ]", "[ REGRESSION ]", "[ PROJECTION ]"])

    with tab1:
        st.markdown(f"#### CORE DATASET: {len(df)} OBSERVATIONS DETECTED")
        st.dataframe(df[selected_vars].head(10), use_container_width=True)
        
        st.markdown("#### TEMPORAL DISTRIBUTION")
        fig = go.Figure()
        for v in selected_vars:
            fig.add_trace(go.Scatter(x=df.index, y=df[v], name=v, line=dict(width=2)))
        fig.update_layout(template="plotly_dark", font_family="Roboto Mono", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### AUGMENTED DICKEY-FULLER ANALYSIS")
        ur_target = st.selectbox("Select Target Variable", selected_vars)
        if st.button("EXECUTE TEST"):
            res = adfuller(df[ur_target].dropna())
            
            # Professional Results Display
            r1, r2, r3 = st.columns(3)
            r1.metric("ADF STATISTIC", f"{res[0]:.4f}")
            r2.metric("P-VALUE", f"{res[1]:.4f}")
            r3.metric("STATUS", "STATIONARY" if res[1] < sig_level else "NON-STATIONARY")

    with tab3:
        st.markdown("#### OLS ESTIMATION")
        y = st.selectbox("Dependent (Y)", selected_vars)
        x = st.multiselect("Exogenous (X)", [v for v in selected_vars if v != y])
        
        if st.button("RUN ESTIMATION") and x:
            model = OLS(df[y], add_constant(df[x])).fit()
            st.code(model.summary().as_text(), language='text')

    with tab4:
        st.markdown("#### ARIMA PROJECTION")
        f_target = st.selectbox("Forecast Variable", selected_vars, key="f")
        h = st.slider("Horizon", 1, 10, 5)
        
        if st.button("GENERATE FORECAST"):
            model = ARIMA(df[f_target], order=(1,1,1)).fit()
            fc = model.forecast(steps=h)
            st.markdown("##### FORECASTED VALUES")
            st.table(pd.DataFrame({"Period": np.arange(1,h+1), "Value": fc}))

else:
    st.markdown("<div style='text-align:center; padding-top:100px; color:#94A3B8'>INITIALIZING SYSTEM: PLEASE UPLOAD DATASET</div>", unsafe_allow_html=True)
