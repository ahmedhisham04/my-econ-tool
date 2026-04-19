import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.tools import add_constant

# ─────────────────────────────────────────────────────────────────
# NEXUS DESIGN SYSTEM
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
    
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 2px solid var(--nexus-border); }
    .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748B !important; }
    .stTabs [aria-selected="true"] { color: var(--nexus-accent) !important; border-bottom: 2px solid var(--nexus-accent) !important; }
    
    .nexus-card { background: white; border: 1px solid var(--nexus-border); padding: 1.2rem; border-radius: 2px; margin-bottom: 1rem; }
    .label-mono { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><div class='label-mono'>[ WORKFILE CONTROL ]</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("LOAD DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("SIGNIFICANCE (α)", [0.01, 0.05, 0.10], index=1)
    
    if st.session_state['initialized']:
        if st.button("CLOSE WORKFILE", use_container_width=True):
            st.session_state['initialized'] = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# WORKSPACE LOGIC
# ─────────────────────────────────────────────────────────────────
if uploaded_file and not st.session_state['initialized']:
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Date Cleaning: Preserve the display but force numbers for math later
        st.markdown("### Workfile Initialization")
        c1, c2 = st.columns([2, 1])
        with c1:
            time_idx = st.selectbox("Temporal Anchor (Date/Observation)", options=df_raw.columns)
            
            # Smart Date Formatting (Preventing 2024.0)
            df = df_raw.dropna(subset=[time_idx]).sort_values(by=time_idx)
            # If it's a year, show as integer
            if df[time_idx].dtype == 'float64' or df[time_idx].dtype == 'int64':
                df['display_date'] = df[time_idx].astype(int).astype(str)
            else:
                df['display_date'] = df[time_idx].astype(str)

            potential_vars = [c for c in df.columns if c not in [time_idx, 'display_date']]
            dep_var = st.selectbox("Dependent Variable (Y)", options=potential_vars)
            indep_vars = st.multiselect("Independent Variables (X)", options=[v for v in potential_vars if v != dep_var])
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if time_idx and dep_var and indep_vars:
                st.markdown(f"""<div class="nexus-card"><div class="label-mono">Observations</div><div style="font-size:20px; font-weight:600;">{len(df)} Points</div></div>""", unsafe_allow_html=True)
                if st.button("OPEN WORKFILE", use_container_width=True):
                    # Final numeric conversion of variables before saving state
                    for v in [dep_var] + indep_vars:
                        df[v] = pd.to_numeric(df[v], errors='coerce')
                    df = df.dropna(subset=[dep_var] + indep_vars)
                    st.session_state.update({"initialized": True, "df": df, "time": time_idx, "y": dep_var, "x": indep_vars})
                    st.rerun()
    except Exception as e:
        st.error(f"Loader Error: {e}")

elif st.session_state['initialized']:
    df, time_idx, dep_var, indep_vars = st.session_state['df'], st.session_state['time'], st.session_state['y'], st.session_state['x']
    
    st.markdown(f"### Research Workspace: {dep_var}")
    tabs = st.tabs(["[ SUMMARY ]", "[ VISUALS ]", "[ STATIONARITY ]", "[ ESTIMATION ]", "[ DIAGNOSTICS ]"])

    with tabs[0]:
        stats = df[[dep_var] + indep_vars].describe().T
        stats['Skew'] = df[[dep_var] + indep_vars].skew()
        stats['Kurtosis'] = df[[dep_var] + indep_vars].kurtosis()
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        st.markdown("<div class='label-mono'>[ CORRELATION MATRIX ]</div>", unsafe_allow_html=True)
        st.dataframe(df[[dep_var] + indep_vars].corr().style.background_gradient(cmap='RdBu_r').format("{:.4f}"), use_container_width=True)

    with tabs[1]:
        mode = st.toggle("Standardize (Z-Score)")
        plot_df = df.copy()
        if mode:
            for v in [dep_var] + indep_vars:
                plot_df[v] = (plot_df[v] - plot_df[v].mean()) / plot_df[v].std()
        
        # FIXED: Visual Date Format using 'display_date'
        fig = px.line(plot_df, x='display_date', y=[dep_var] + indep_vars, template="plotly_white")
        fig.update_layout(font_family="Inter", hovermode="x unified", xaxis_title=time_idx, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        for var in [dep_var] + indep_vars:
            res = adfuller(df[var].dropna())
            color = "#10B981" if res[1] < sig_level else "#F59E0B"
            st.markdown(f"<div style='border-left: 4px solid {color}; padding: 10px; background: white; border: 1px solid #E2E8F0; margin-bottom:5px;'><b>{var}</b>: {'STATIONARY' if res[1] < sig_level else 'NON-STATIONARY'} (p={res[1]:.4f})</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='label-mono'>[ ESTIMATION ENGINE ]</div>", unsafe_allow_html=True)
        method = st.radio("Technique", ["Ordinary Least Squares (OLS)", "Autoregressive Distributed Lag (ARDL)"], horizontal=True)
        
        if "OLS" in method:
            use_const = st.checkbox("Include Constant (β₀)", value=True)
            # FULL LaTeX Specification
            formula = f"{dep_var}_t = "
            if use_const: formula += "\\beta_0 + "
            formula += " + ".join([f"\\beta_{i+1} \\text{{{v}}}_t" for i, v in enumerate(indep_vars)]) + " + \\epsilon_t"
            st.latex(formula)
            
            if st.button("RUN OLS ESTIMATION", use_container_width=True):
                Y, X = df[dep_var], df[indep_vars]
                if use_const: X = add_constant(X)
                st.session_state['model_res'] = sm.OLS(Y, X).fit()

        elif "ARDL" in method:
            st.markdown("<div class='label-mono'>[ ARDL SPECIFICATION ]</div>", unsafe_allow_html=True)
            st.latex(f"{dep_var}_t = c + \\sum_{{i=1}}^{{p}} \\alpha_i {dep_var}_{{t-i}} + \\sum_{{j=0}}^{{q}} \\beta_j X_{{t-j}} + \\epsilon_t")
            max_lag = st.number_input("Max Lag Selection (p, q)", min_value=1, max_value=4, value=1)
            
            if st.button("RUN ARDL ESTIMATION", use_container_width=True):
                try:
                    # Automatic lag selection using AIC
                    sel_res = ardl_select_order(df[dep_var], max_lag, df[indep_vars], max_lag, ic='aic', trend='c')
                    model = ARDL(df[dep_var], sel_res.model_order['dep'], df[indep_vars], sel_res.model_order['exog'], trend='c').fit()
                    st.session_state['model_res'] = model
                    st.success(f"Optimal Lags Selected: {sel_res.model_order}")
                except Exception as e:
                    st.error(f"ARDL Estimation failed: {e}")

        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            st.markdown("<div class='label-mono'>[ ESTIMATION OUTPUT ]</div>", unsafe_allow_html=True)
            # Handle different summary types between OLS and ARDL
            if hasattr(res, 'params'):
                out_df = pd.DataFrame({"Coef.": res.params, "Std.Err": res.bse, "t-Stat": res.tvalues, "Prob.": res.pvalues})
                st.table(out_df.style.format("{:.4f}"))

    with tabs[4]:
        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            st.markdown("<div class='label-mono'>[ DIAGNOSTIC TESTS ]</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig_res = px.line(x=df['display_date'], y=res.resid, title="Residuals", template="plotly_white")
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)
            with c2:
                jb_stat, jb_p, _, _ = jarque_bera(res.resid)
                st.markdown(f"""
                <div class="nexus-card">
                    <div class="label-mono">Metrics</div>
                    Durbin-Watson: <b>{durbin_watson(res.resid):.2f}</b><br>
                    Jarque-Bera (p): <b>{jb_p:.4f}</b> {'✅' if jb_p > 0.05 else '❌'}
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center; padding: 100px; color:#94A3B8;'>AWAITING DATASET...</div>", unsafe_allow_html=True)
