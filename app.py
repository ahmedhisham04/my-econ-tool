import warnings, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from statsmodels.tsa.stattools   import adfuller, kpss, acf, pacf
from statsmodels.tsa.ardl        import ARDL
from statsmodels.tsa.api         import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools     import add_constant
from statsmodels.stats.stattools  import durbin_watson
from statsmodels.stats.diagnostic import (het_arch, acorr_ljungbox,
                                           het_breuschpagan)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# PAGE CONFIG
st.set_page_config(
    page_title="Macro-Financial Analytics Suite",
    page_icon="🏦",
    layout="wide",
)

# THEME DEFINITIONS
THEMES = {
    "Executive Dark": {
        "bg": "#080E1A", "card": "#0F1C30", "text": "#E8F0FE", "accent": "#00C9A7", "plotly": "plotly_dark"
    },
    "Research Light": {
        "bg": "#F4F6FA", "card": "#FFFFFF", "text": "#1A2233", "accent": "#1A56DB", "plotly": "plotly_white"
    },
}

with st.sidebar:
    st.markdown("### 🏦 MF Analytics Suite")
    theme_name = st.selectbox("🎨 Theme", list(THEMES.keys()))
    T = THEMES[theme_name]
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    sig_level = st.select_slider("Significance Level α", options=[0.01, 0.05, 0.10], value=0.05)

# HELPER: FIXED KPI CARD (The part that caused the crash)
def kpi_card(label, value, sub="", accent="accent-left", delta=None):
    delta_html = ""
    if delta is not None and isinstance(delta, (int, float)):
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        delta_html = f'<div class="{cls}">{"▲" if delta>=0 else "▼"} {abs(delta):.2f}</div>'
    
    return f"""
    <div class="mf-card {accent}">
      <div style="font-size:0.7rem; color:gray; text-transform:uppercase;">{label}</div>
      <div style="font-size:1.5rem; font-weight:bold; color:{T['text']};">{value}</div>
      <div style="font-size:0.75rem; color:gray;">{sub}</div>
      {delta_html}
    </div>"""

# CSS Injection
st.markdown(f"""
<style>
    .stApp {{ background-color: {T['bg']}; color: {T['text']}; }}
    .mf-card {{ 
        background: {T['card']}; padding: 20px; border-radius: 10px; 
        border-left: 5px solid {T['accent']}; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); 
        margin-bottom: 10px;
    }}
    .accent-gold {{ border-left: 5px solid #FFB800; }}
    .accent-danger {{ border-left: 5px solid #FF5370; }}
    .kpi-delta-pos {{ color: #00C9A7; }}
    .kpi-delta-neg {{ color: #FF5370; }}
</style>
""", unsafe_allow_html=True)

if uploaded:
    # Aggressive Cleaning for the "Million Row" problem
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    
    # 1. Drop completely empty rows/cols
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # 2. Keep only rows where at least half the columns have data (kills metadata/footnotes)
    df = df.dropna(thresh=len(df.columns) // 2).reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Workspace", "🔬 Tests", "📐 Models", "🩺 Diagnostics", "🔮 Forecast"])

    with tab1:
        st.markdown("### Dataset Overview")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(kpi_card("Observations", f"{len(df):,}", "rows detected"), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Variables", str(len(numeric_cols)), "numeric columns", accent="accent-gold"), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Quality", "Cleaned", "Empty rows removed"), unsafe_allow_html=True)
        
        st.write("### Data Preview")
        st.dataframe(df.head(10))

        if len(numeric_cols) >= 1:
            sel_var = st.selectbox("Select variable to plot", numeric_cols)
            fig = px.line(df, y=sel_var, template=T['plotly'])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Unit Root Test (ADF)")
        test_var = st.selectbox("Variable for ADF", numeric_cols)
        if st.button("Run ADF"):
            res = adfuller(df[test_var].dropna())
            st.write(f"ADF Statistic: {res[0]:.4f}")
            st.write(f"p-value: {res[1]:.4f}")
            if res[1] < sig_level:
                st.success("Stationary (I(0))")
            else:
                st.warning("Non-Stationary - Try Differencing")

    with tab3:
        st.write("### OLS Regression")
        y_var = st.selectbox("Dependent Variable (Y)", numeric_cols)
        x_vars = st.multiselect("Independent Variables (X)", [c for c in numeric_cols if c != y_var])
        if st.button("Run OLS") and x_vars:
            Y = df[y_var]
            X = add_constant(df[x_vars])
            model = OLS(Y, X).fit()
            st.text(model.summary())

    with tab4:
        st.write("Diagnostics available after running OLS in Tab 3.")

    with tab5:
        st.write("### ARIMA Forecasting")
        f_var = st.selectbox("Forecast Variable", numeric_cols, key="f")
        steps = st.number_input("Steps to forecast", 1, 50, 5)
        if st.button("Generate Forecast"):
            model = ARIMA(df[f_var], order=(1,1,1)).fit()
            fc = model.forecast(steps=steps)
            st.write(fc)
            st.line_chart(fc)

else:
    st.info("Please upload your data file to begin.")
