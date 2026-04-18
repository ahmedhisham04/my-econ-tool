import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import add_constant

# ─────────────────────────────────────────────────────────────────
# INSTITUTIONAL BRANDING & CSS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CBE Macro-Financial Workbench", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@700&display=swap');

    /* Color Palette */
    :root {
        --cbe-navy: #002B49;
        --cbe-gold: #C5A059;
        --cbe-slate: #F8FAFC;
        --cbe-border: #E2E8F0;
    }

    .stApp { background-color: white; color: #1E293B; font-family: 'Inter', sans-serif; }

    /* Central Bank Header */
    .cbe-header {
        background-color: var(--cbe-navy);
        padding: 2rem;
        border-bottom: 5px solid var(--cbe-gold);
        margin-bottom: 2rem;
        color: white;
    }
    .cbe-title { font-family: 'Playfair Display', serif; font-size: 28px; letter-spacing: 1px; }
    .cbe-subtitle { font-size: 14px; opacity: 0.8; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] { background-color: var(--cbe-navy) !important; color: white !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    
    /* Institutional Cards */
    .executive-card {
        border: 1px solid var(--cbe-border);
        border-radius: 0px; /* Sharp corners for professional look */
        padding: 20px;
        background-color: var(--cbe-slate);
        border-top: 4px solid var(--cbe-navy);
        margin-bottom: 20px;
    }
    .metric-label { font-size: 12px; color: #64748B; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 24px; color: var(--cbe-navy); font-weight: 700; }

    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #64748B; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: var(--cbe-navy) !important; border-bottom: 2px solid var(--cbe-gold) !important; }

    /* Form Inputs */
    .stButton>button {
        background-color: var(--cbe-navy);
        color: white;
        border-radius: 0px;
        border: 1px solid var(--cbe-gold);
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover { background-color: var(--cbe-gold); color: var(--cbe-navy); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TOP BANNER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cbe-header">
    <div class="cbe-title">CENTRAL BANK OF EGYPT</div>
    <div class="cbe-subtitle">Macro-Financial Research & Analytics Division</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA IMPORT & WIZARD
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ARCHIVE DATA IMPORT")
    uploaded = st.file_uploader("Upload Institutional Data (CSV/XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)

if uploaded:
    # Load Data
    if uploaded.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)

    # INITIAL CLEANING
    df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')

    st.markdown("### DATA CONFIGURATION PROTOCOL")
    st.info("The system requires a defined Temporal Index to initialize analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        time_col = st.selectbox("Select Temporal Column (Year)", df_raw.columns)
    with col2:
        freq = st.selectbox("Frequency of Observations", ["Annual", "Quarterly", "Monthly"])

    # ENFORCED CLEANING: Filter dataset based on valid Year entries
    # This specifically fixes the "Million Row" and "Variables as Year" problem
    df = df_raw.dropna(subset=[time_col]).copy()
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(by=time_col)
    df = df.set_index(time_col)
    
    # Identify variables (excluding index)
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()

    # ─────────────────────────────────────────────────────────────────
    # EXECUTIVE DASHBOARD
    # ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["STATISTICAL SUMMARY", "STATIONARITY", "ECONOMETRIC MODELS", "DIAGNOSTICS", "PROJECTION"])

    with tabs[0]:
        st.markdown("#### Sample Characteristics")
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f'<div class="executive-card"><div class="metric-label">Observations</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="executive-card"><div class="metric-label">Date Range</div><div class="metric-value">{int(df.index.min())} - {int(df.index.max())}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="executive-card"><div class="metric-label">Variables</div><div class="metric-value">{len(numeric_vars)}</div></div>', unsafe_allow_html=True)
        
        st.write("#### Data Verification Table")
        st.dataframe(df.head(10), use_container_width=True)

    with tabs[1]:
        st.markdown("#### Augmented Dickey-Fuller (ADF) Unit Root Test")
        target = st.selectbox("Target Variable", numeric_vars)
        if st.button("RUN ANALYSIS"):
            res = adfuller(df[target].dropna())
            decision = "STATIONARY" if res[1] < sig_level else "NON-STATIONARY"
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="executive-card"><div class="metric-label">p-Value</div><div class="metric-value">{res[1]:.4f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="executive-card"><div class="metric-label">Verdict</div><div class="metric-value">{decision}</div></div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("#### Model Estimation")
        model_choice = st.selectbox("Specification Type", ["Ordinary Least Squares (OLS)", "VAR"])
        
        y_var = st.selectbox("Dependent Variable (Y)", numeric_vars)
        x_vars = st.multiselect("Exogenous Variables (X)", [v for v in numeric_vars if v != y_var])
        
        if st.button("EXECUTE REGRESSION") and x_vars:
            Y = df[y_var]
            X = add_constant(df[x_vars])
            results = OLS(Y, X).fit()
            st.text(results.summary())

    with tabs[4]:
        st.markdown("#### ARIMA Projection")
        f_var = st.selectbox("Variable to Forecast", numeric_vars, key="f")
        h = st.slider("Forecast Horizon (Years)", 1, 10, 5)
        
        if st.button("GENERATE PROJECTION"):
            model = ARIMA(df[f_var], order=(1,1,1)).fit()
            fc = model.forecast(steps=h)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[f_var], name="Historical", line=dict(color="#002B49")))
            fig.add_trace(go.Scatter(x=np.arange(df.index.max()+1, df.index.max()+1+h), y=fc, name="Forecast", line=dict(color="#C5A059", dash="dash")))
            fig.update_layout(template="plotly_white", font_family="Inter", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <h1 style='color: #002B49; font-family: Playfair Display;'>SYSTEM INITIALIZATION</h1>
        <p style='color: #64748B;'>Please upload institutional data files to proceed with analysis.</p>
    </div>
    """, unsafe_allow_html=True)
