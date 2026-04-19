import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson, jarque_bera
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
    # PHASE 1: INITIALIZATION
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        st.markdown("### Workfile Initialization")
        c1, c2 = st.columns([2, 1])
        with c1:
            time_idx = st.selectbox("Temporal Anchor (Date/Obs)", options=df_raw.columns)
            df = df_raw.dropna(subset=[time_idx]).sort_values(by=time_idx)
            potential_vars = [c for c in df.columns if c != time_idx]
            dep_var = st.selectbox("Dependent Variable (Y)", options=potential_vars)
            indep_vars = st.multiselect("Independent Variables (X)", options=[v for v in potential_vars if v != dep_var])
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if time_idx and dep_var and indep_vars:
                st.markdown(f"""<div class="nexus-card"><div class="label-mono">Observations</div><div style="font-size:20px; font-weight:600;">{len(df)} Points</div></div>""", unsafe_allow_html=True)
                if st.button("OPEN WORKFILE", use_container_width=True):
                    st.session_state.update({"initialized": True, "df": df, "time": time_idx, "y": dep_var, "x": indep_vars})
                    st.rerun()
    except Exception as e:
        st.error(f"Loader Error: {e}")

elif st.session_state['initialized']:
    # PHASE 2: RESEARCH KERNEL
    df, time_idx, dep_var, indep_vars = st.session_state['df'], st.session_state['time'], st.session_state['y'], st.session_state['x']
    
    st.markdown(f"### Research Workspace: {dep_var}")
    tabs = st.tabs(["[ SUMMARY ]", "[ VISUALS ]", "[ STATIONARITY ]", "[ ESTIMATION ]", "[ DIAGNOSTICS ]"])

    with tabs[0]:
        stats = df[[dep_var] + indep_vars].describe().T
        stats['Skew'] = df[[dep_var] + indep_vars].skew()
        stats['Kurtosis'] = df[[dep_var] + indep_vars].kurtosis()
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        st.markdown("<div class='label-mono'>[ CORRELATION HEATMAP ]</div>", unsafe_allow_html=True)
        st.dataframe(df[[dep_var] + indep_vars].corr().style.background_gradient(cmap='RdBu_r').format("{:.4f}"), use_container_width=True)

    with tabs[1]:
        mode = st.toggle("Standardize Variables (Z-Score)", help="Useful when variables have different units (e.g. GDP vs %)")
        plot_df = df.copy()
        if mode:
            for v in [dep_var] + indep_vars:
                plot_df[v] = (plot_df[v] - plot_df[v].mean()) / plot_df[v].std()
        
        fig = px.line(plot_df, x=time_idx, y=[dep_var] + indep_vars, template="plotly_white")
        fig.update_layout(font_family="Inter", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        for var in [dep_var] + indep_vars:
            res = adfuller(df[var].dropna())
            is_stat = res[1] < sig_level
            color = "#10B981" if is_stat else "#F59E0B"
            st.markdown(f"<div style='border-left: 4px solid {color}; padding: 10px; background: white; border: 1px solid #E2E8F0; margin-bottom:5px;'><b>{var}</b>: {'STATIONARY' if is_stat else 'NON-STATIONARY'} (p={res[1]:.4f})</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='label-mono'>[ MODEL SPECIFICATION ]</div>", unsafe_allow_html=True)
        use_const = st.checkbox("Include Constant ($\beta_0$)", value=True)
        
        # OLS Logic
        st.latex(f"{dep_var}_t = {'\\beta_0 +' if use_const else ''} " + " + ".join([f"\\beta_{i+1} {v}_t" for i, v in enumerate(indep_vars)]) + " + \\epsilon_t")
        if st.button("RUN OLS ESTIMATION", use_container_width=True):
            Y = df[dep_var]
            X = df[indep_vars]
            if use_const: X = add_constant(X)
            model = sm.OLS(Y, X).fit()
            st.session_state['model_res'] = model

        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            out_df = pd.DataFrame({"Coef.": res.params, "Std.Err": res.bse, "t-Stat": res.tvalues, "Prob.": res.pvalues})
            st.table(out_df.style.format("{:.4f}"))
            c1, c2, c3 = st.columns(3)
            c1.metric("R-squared", f"{res.rsquared:.4f}")
            c2.metric("Adj R-sq", f"{res.rsquared_adj:.4f}")
            c3.metric("F-Stat Prob.", f"{res.f_pvalue:.4f}")

    with tabs[4]:
        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            col_a, col_b = st.columns(2)
            with col_a:
                fig_res = px.line(x=df[time_idx], y=res.resid, title="Residuals Plot", template="plotly_white")
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)
            with col_b:
                fig_hist = px.histogram(res.resid, nbins=20, title="Error Distribution", template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            jb_stat, jb_p, _, _ = jarque_bera(res.resid)
            st.markdown(f"""
            <div class="nexus-card">
                <div class="label-mono">Diagnostic Tests</div>
                Durbin-Watson (Autocorrelation): <b>{durbin_watson(res.resid):.2f}</b><br>
                Jarque-Bera (Normality): <b>{jb_p:.4f}</b> {'✅' if jb_p > 0.05 else '❌'}
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center; padding: 100px; color:#94A3B8;'>AWAITING WORKFILE...</div>", unsafe_allow_html=True)
