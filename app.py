import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats

from statsmodels.tsa.stattools   import adfuller, kpss
from statsmodels.tsa.ardl        import ARDL
from statsmodels.tsa.api         import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools     import add_constant

# ─────────────────────────────────────────────────────────────────
# SYSTEM CONFIGURATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro-Financial Suite | Research Edition", layout="wide")

# PROFESSIONAL STYLING (The "Stata/EViews" Look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Mono&display=swap');
    
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Headers */
    .app-title { font-size: 24px; font-weight: 600; color: #FFFFFF; border-bottom: 2px solid #1F2937; padding-bottom: 10px; margin-bottom: 20px; }
    .section-header { font-family: 'IBM Plex Mono', monospace; font-size: 14px; text-transform: uppercase; color: #9CA3AF; margin-top: 25px; border-left: 3px solid #3B82F6; padding-left: 10px; }
    
    /* Research Cards */
    .research-card { background: #161B22; border: 1px solid #30363D; border-radius: 4px; padding: 15px; margin-bottom: 15px; }
    .stat-label { font-size: 12px; color: #8B949E; text-transform: uppercase; }
    .stat-value { font-size: 20px; font-weight: 600; color: #58A6FF; font-family: 'IBM Plex Mono', monospace; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0D1117 !important; border-right: 1px solid #30363D; }
    
    /* Tables */
    .stDataFrame { border: 1px solid #30363D; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='app-title'>MF ANALYTICS SUITE</div>", unsafe_allow_html=True)
    st.markdown("SYSTEM STATUS: **READY**")
    uploaded = st.file_uploader("IMPORT DATASET (CSV/XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.select_slider("SIGNIFICANCE ALPHA", options=[0.01, 0.05, 0.10], value=0.05)

# ─────────────────────────────────────────────────────────────────
# DATA CONFIGURATION WIZARD
# ─────────────────────────────────────────────────────────────────
if uploaded:
    if uploaded.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
    
    # 1. First-pass cleaning (remove purely empty space)
    df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')

    st.markdown("<div class='section-header'>Step 1: Data Configuration</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        time_col = st.selectbox("Select Time Index (Year/Date)", df_raw.columns)
    with c2:
        freq = st.selectbox("Frequency", ["Annual", "Quarterly", "Monthly"])

    # Final Research-Grade Cleaning
    df = df_raw.dropna(subset=[time_col]).reset_index(drop=True)
    df[time_col] = df[time_col].astype(int) # Forces clean year integers
    df = df.set_index(time_col)
    
    # Identify variables (excluding index)
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown(f"**CONSOLIDATED SAMPLE:** {df.index.min()} to {df.index.max()} | **OBSERVATIONS:** {len(df)}")

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS INTERFACE
    # ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["WORKSPACE", "STATIONARITY", "ECONOMETRIC MODELS", "DIAGNOSTICS", "FORECAST"])

    with tabs[0]:
        st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df[numeric_vars].describe().T, use_container_width=True)
        
        st.markdown("<div class='section-header'>Visual Inspection</div>", unsafe_allow_html=True)
        sel_vars = st.multiselect("Select Variables to Plot", numeric_vars, default=numeric_vars[:1])
        if sel_vars:
            fig = px.line(df[sel_vars], template="plotly_dark")
            fig.update_layout(font_family="IBM Plex Mono", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("<div class='section-header'>Unit Root Analysis</div>", unsafe_allow_html=True)
        target_ur = st.selectbox("Variable", numeric_vars, key="ur")
        
        if st.button("EXECUTE ADF TEST"):
            res = adfuller(df[target_ur].dropna())
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='research-card'><div class='stat-label'>ADF Statistic</div><div class='stat-value'>{res[0]:.4f}</div></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='research-card'><div class='stat-label'>p-value</div><div class='stat-value'>{res[1]:.4f}</div></div>", unsafe_allow_html=True)
            with c3:
                decision = "STATIONARY" if res[1] < sig_level else "NON-STATIONARY"
                st.markdown(f"<div class='research-card'><div class='stat-label'>Verdict</div><div class='stat-value'>{decision}</div></div>", unsafe_allow_html=True)

    with tabs[2]:
        model_type = st.radio("Specification", ["OLS", "ARDL", "VAR"], horizontal=True)
        
        if model_type == "OLS":
            st.markdown("<div class='section-header'>Ordinary Least Squares</div>", unsafe_allow_html=True)
            st.latex(r"y_t = \beta_0 + \beta_1 X_{1,t} + \dots + \beta_k X_{k,t} + \epsilon_t")
            
            dep = st.selectbox("Dependent (Y)", numeric_vars)
            indep = st.multiselect("Independent (X)", [v for v in numeric_vars if v != dep])
            
            if st.button("ESTIMATE MODEL") and indep:
                Y = df[dep]
                X = add_constant(df[indep])
                res = OLS(Y, X).fit()
                st.text(res.summary())

    with tabs[4]:
        st.markdown("<div class='section-header'>ARIMA Forecasting</div>", unsafe_allow_html=True)
        f_var = st.selectbox("Forecast Target", numeric_vars, key="f")
        h = st.number_input("Horizon (Periods)", 1, 10, 5)
        
        if st.button("RUN FORECAST"):
            model = ARIMA(df[f_var], order=(1,1,1)).fit()
            forecast = model.forecast(steps=h)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df.index, y=df[f_var], name="Actual"))
            fig_f.add_trace(go.Scatter(x=np.arange(df.index.max()+1, df.index.max()+1+h), y=forecast, name="Forecast", line=dict(dash='dash')))
            fig_f.update_layout(template="plotly_dark", font_family="IBM Plex Mono")
            st.plotly_chart(fig_f, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <h2 style='color: #58A6FF;'>MACRO-FINANCIAL ANALYTICS SUITE</h2>
        <p style='color: #8B949E;'>Academic Research Environment | Version 2.1</p>
        <p>Please import a valid time-series dataset to initialize the workbench.</p>
    </div>
    """, unsafe_allow_html=True)
