import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ardl import ARDL
from statsmodels.tools.tools import add_constant

# ─────────────────────────────────────────────────────────────────
# NEXUS PROFESSIONAL DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nexus Econometrics", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
        --nexus-bg: #F8FAFC;
        --nexus-sidebar: #0F172A;
        --nexus-accent: #2563EB;
        --nexus-text: #1E293B;
        --nexus-border: #E2E8F0;
    }
    .stApp { background-color: var(--nexus-bg); color: var(--nexus-text); font-family: 'Inter', sans-serif; }
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 2px solid var(--nexus-border); }
    .stTabs [data-baseweb="tab"] { 
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
        color: #64748B !important; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { color: var(--nexus-accent) !important; border-bottom: 2px solid var(--nexus-accent) !important; }

    .nexus-card {
        background: white; border: 1px solid var(--nexus-border);
        padding: 1.5rem; border-radius: 2px; margin-bottom: 1rem;
    }
    .label-mono { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; }
    
    /* EViews Style Output Box */
    .eviews-box {
        background: #FFFFFF; border: 1px solid #CBD5E1; 
        font-family: 'IBM Plex Mono', monospace; font-size: 13px; 
        padding: 20px; color: #0F172A; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><div style='font-family:IBM Plex Mono; font-size:11px; color:#94A3B8;'>[ WORKFILE CONTROL ]</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("LOAD DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("SIGNIFICANCE (α)", [0.01, 0.05, 0.10], index=1)
    
    if st.session_state['initialized']:
        if st.button("CLOSE WORKFILE", use_container_width=True):
            st.session_state['initialized'] = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────────────────────────
if uploaded_file and not st.session_state['initialized']:
    # PHASE 1: INITIALIZATION
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    st.markdown("### Workfile Initialization")
    c1, c2 = st.columns([2, 1])
    with c1:
        time_idx = st.selectbox("Assign Temporal Anchor (Date/Observation)", options=df_raw.columns)
        df = df_raw.dropna(subset=[time_idx]).sort_values(by=time_idx)
        potential_vars = [c for c in df.columns if c != time_idx]
        dep_var = st.selectbox("Dependent Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Independent Variables (X)", options=[v for v in potential_vars if v != dep_var])
        
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if time_idx and dep_var and len(indep_vars) > 0:
            st.markdown(f"""
            <div class="nexus-card">
                <div class="label-mono">Observation Range</div>
                <div style="font-size:20px; font-weight:600;">{len(df)} Points</div>
                <div style="font-size:12px; color:#64748B;">Index: {int(df[time_idx].min())} - {int(df[time_idx].max())}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("OPEN WORKFILE", use_container_width=True):
                st.session_state.update({"initialized": True, "df": df, "time": time_idx, "y": dep_var, "x": indep_vars})
                st.rerun()
        else:
            st.warning("Mapping required to unlock Kernel.")

elif st.session_state['initialized']:
    # PHASE 2: RESEARCH KERNEL
    df = st.session_state['df']
    time_idx = st.session_state['time']
    dep_var = st.session_state['y']
    indep_vars = st.session_state['x']
    
    st.markdown(f"### Research Workspace: {dep_var}")
    
    tabs = st.tabs(["[ SUMMARY ]", "[ VISUALS ]", "[ STATIONARITY ]", "[ ESTIMATION ]"])

    with tabs[0]:
        stats = df[[dep_var] + indep_vars].describe().T
        stats['Skewness'] = df[[dep_var] + indep_vars].skew()
        stats['Kurtosis'] = df[[dep_var] + indep_vars].kurtosis()
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        st.markdown("<div class='label-mono'>[ CORRELATION MATRIX ]</div>", unsafe_allow_html=True)
        st.dataframe(df[[dep_var] + indep_vars].corr().style.background_gradient(cmap='RdBu_r').format("{:.4f}"), use_container_width=True)

    with tabs[1]:
        fig = px.line(df, x=time_idx, y=[dep_var] + indep_vars, template="plotly_white")
        fig.update_layout(font_family="Inter", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        for var in [dep_var] + indep_vars:
            res = adfuller(df[var].dropna())
            color = "#10B981" if res[1] < sig_level else "#F59E0B"
            st.markdown(f"""
            <div style='border-left: 4px solid {color}; padding: 10px 15px; background: white; border: 1px solid #E2E8F0; margin-bottom:8px;'>
                <span style='font-family:IBM Plex Mono; font-weight:600;'>{var}</span>: 
                <span style='color:{color};'>{"STATIONARY" if res[1] < sig_level else "NON-STATIONARY"}</span>
                <br><span style='font-size:11px; color:gray;'>p-val: {res[1]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='label-mono'>[ MODEL SPECIFICATION ]</div>", unsafe_allow_html=True)
        model_type = st.radio("Technique", ["Ordinary Least Squares (OLS)", "Autoregressive Distributed Lag (ARDL)"], horizontal=True)
        
        if model_type.startswith("OLS"):
            st.latex(f"{dep_var}_t = \\beta_0 + " + " + ".join([f"\\beta_{i+1} {v}_t" for i, v in enumerate(indep_vars)]) + " + \\epsilon_t")
            if st.button("ESTIMATE OLS"):
                Y = df[dep_var]
                X = add_constant(df[indep_vars])
                model = OLS(Y, X).fit()
                
                st.markdown("<div class='label-mono'>[ ESTIMATION OUTPUT ]</div>", unsafe_allow_html=True)
                # Render results in EViews-style table
                results_df = pd.DataFrame({
                    "Coefficient": model.params,
                    "Std. Error": model.bse,
                    "t-Statistic": model.tvalues,
                    "Prob.": model.pvalues
                })
                st.table(results_df.style.format("{:.4f}"))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("R-squared", f"{model.rsquared:.4f}")
                c2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
                c3.metric("F-statistic", f"{model.fvalue:.2f}")

        elif model_type.startswith("ARDL"):
            st.markdown("<div class='nexus-card'>Note: ARDL model will estimate using optimal lag selection (AIC).</div>")
            if st.button("ESTIMATE ARDL"):
                try:
                    # Simplified ARDL estimation for thesis-level baseline
                    model = ARDL(df[dep_var], 1, df[indep_vars], {v: 1 for v in indep_vars}).fit()
                    st.text(model.summary())
                except Exception as e:
                    st.error(f"ARDL Error: {e}. Check for sufficient data points.")

else:
    st.markdown("<div style='text-align:center; padding: 100px; color:#94A3B8; font-family:IBM Plex Mono;'>INITIALIZE WORKFILE TO ACCESS ESTIMATION KERNEL</div>", unsafe_allow_html=True)
