import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
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
    
    /* Workspace Tabs (EViews Style) */
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
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        st.markdown("### Workfile Initialization")
        c1, c2 = st.columns([2, 1])
        with c1:
            time_idx = st.selectbox("Select Temporal Anchor (Date/Observation)", options=df_raw.columns)
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
                    <div style="font-size:12px; color:#64748B;">{time_idx}: {df[time_idx].iloc[0]} to {df[time_idx].iloc[-1]}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("OPEN WORKFILE", use_container_width=True):
                    st.session_state.update({"initialized": True, "df": df, "time": time_idx, "y": dep_var, "x": indep_vars})
                    st.rerun()
            else:
                st.warning("Mapping required to initialize.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

elif st.session_state['initialized']:
    # PHASE 2: RESEARCH KERNEL
    df = st.session_state['df']
    time_idx = st.session_state['time']
    dep_var = st.session_state['y']
    indep_vars = st.session_state['x']
    
    st.markdown(f"### Workfile: {dep_var} Analysis")
    
    tabs = st.tabs(["[ SUMMARY ]", "[ VISUALS ]", "[ STATIONARITY ]", "[ ESTIMATION ]", "[ DIAGNOSTICS ]"])

    with tabs[0]:
        st.markdown("<div class='label-mono'>[ DESCRIPTIVE STATISTICS ]</div>", unsafe_allow_html=True)
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
            is_stat = res[1] < sig_level
            color = "#10B981" if is_stat else "#F59E0B"
            st.markdown(f"""
            <div style='border-left: 4px solid {color}; padding: 10px 15px; background: white; border: 1px solid #E2E8F0; margin-bottom:8px;'>
                <span style='font-family:IBM Plex Mono; font-weight:600;'>{var}</span>: 
                <span style='color:{color}; font-weight:600;'>{"STATIONARY" if is_stat else "NON-STATIONARY"}</span>
                <br><span style='font-size:11px; color:gray;'>p-val: {res[1]:.4f} | ADF Stat: {res[0]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='label-mono'>[ ESTIMATION ENGINE ]</div>", unsafe_allow_html=True)
        col_type, col_run = st.columns([3, 1])
        with col_type:
            model_type = st.radio("Technique", ["Ordinary Least Squares (OLS)", "Autoregressive Distributed Lag (ARDL)"], horizontal=True)
        
        # 1. OLS EXECUTION
        if "OLS" in model_type:
            st.latex(f"{dep_var}_t = \\beta_0 + " + " + ".join([f"\\beta_{i+1} {v}_t" for i, v in enumerate(indep_vars)]) + " + \\epsilon_t")
            if st.button("RUN ESTIMATION", use_container_width=True):
                Y = df[dep_var]
                X = add_constant(df[indep_vars])
                model = OLS(Y, X).fit()
                st.session_state['model_res'] = model
                st.success("Model Estimated. See Results below.")

            if 'model_res' in st.session_state:
                res = st.session_state['model_res']
                st.markdown("<div class='label-mono'>[ COEFFICIENT TABLE ]</div>", unsafe_allow_html=True)
                out_df = pd.DataFrame({"Coef.": res.params, "Std.Err": res.bse, "t-Stat": res.tvalues, "Prob.": res.pvalues})
                st.table(out_df.style.format("{:.4f}"))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("R-squared", f"{res.rsquared:.4f}")
                c2.metric("Adj R-sq", f"{res.rsquared_adj:.4f}")
                c3.metric("Durbin-Watson", f"{durbin_watson(res.resid):.2f}")

    with tabs[4]:
        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            st.markdown("<div class='label-mono'>[ RESIDUAL ANALYSIS ]</div>", unsafe_allow_html=True)
            
            # Residual Plot
            resid_df = pd.DataFrame({"Date": df[time_idx], "Residuals": res.resid})
            fig_res = px.line(resid_df, x="Date", y="Residuals", title="Model Residuals (Noise Check)", template="plotly_white")
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Diagnostic Stats
            dw = durbin_watson(res.resid)
            st.markdown(f"""
            <div class="nexus-card">
                <div class="label-mono">Autocorrelation Check</div>
                <div style="font-size:18px;">Durbin-Watson Stat: <b>{dw:.4f}</b></div>
                <div style="font-size:12px; color:#64748B;">
                    (2.0 = No Autocorrelation | < 1.5 = Positive Autocorrelation risk)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run Estimation to unlock Diagnostics.")

else:
    st.markdown("<div style='text-align:center; padding: 100px; color:#94A3B8; font-family:IBM Plex Mono;'>LOAD DATASET TO INITIALIZE WORKFILE</div>", unsafe_allow_html=True)
