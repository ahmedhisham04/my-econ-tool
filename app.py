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
# 01: SYSTEM CORE & AUTHENTICATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nexus Econometrics | Ahmed Hisham", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    _, col_login, _ = st.columns([1, 2, 1])
    with col_login:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #0F172A; font-weight: 700; letter-spacing: -1px;'>NEXUS KERNEL</h1>
            <p style='color: #64748B; font-size: 14px;'>Professional Econometric Research Environment</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div style='background: white; border: 1px solid #E2E8F0; padding: 2rem; border-radius: 4px;'>", unsafe_allow_html=True)
            c_fn, c_ln = st.columns(2)
            first_name = c_fn.text_input("First Name")
            last_name = c_ln.text_input("Last Name")
            email = st.text_input("Email Address")
            occupation = st.selectbox("Occupation", ["Economic Researcher", "Financial Analyst", "Graduate Student", "Academic Professor", "Data Scientist"])
            
            if st.button("LAUNCH WORKBENCH", use_container_width=True):
                if first_name and last_name and email:
                    st.session_state['user_name'] = first_name
                    st.session_state['user_role'] = occupation
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("Credentials required.")
            st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# 02: THE FINGERPRINT (Industrial CSS)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --nexus-bg: #E0E7FF;          /* EViews Light Blue Tint */
        --nexus-sidebar: #0F172A;      /* Midnight Navy */
        --nexus-highlight: #2563EB;    
        --nexus-text: #1E293B;
    }

    .stApp { background-color: var(--nexus-bg); color: var(--nexus-text); font-family: 'Inter', sans-serif; }

    /* Sidebar Branding */
    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; font-family: 'IBM Plex Mono', monospace; }

    /* Professional Workspace Cards */
    .nexus-card { 
        background: white; border: 1px solid #CBD5E1; 
        padding: 1.5rem; border-radius: 2px; 
        box-shadow: 4px 4px 0px 0px rgba(15, 23, 42, 0.1); 
        margin-bottom: 1rem; 
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 2px solid #CBD5E1; }
    .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748B !important; }
    .stTabs [aria-selected="true"] { color: var(--nexus-highlight) !important; border-bottom: 2px solid var(--nexus-highlight) !important; }
    
    .label-mono { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 03: COMMAND CENTER (Sidebar)
# ─────────────────────────────────────────────────────────────────
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

with st.sidebar:
    st.markdown(f"""
    <div style='padding: 1rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 2rem;'>
        <div style='font-size: 16px; font-weight: 700; color: #F8FAFC !important; letter-spacing: 1px;'>NEXUS KERNEL</div>
        <div style='font-size: 10px; color: #64748B !important; font-family: IBM Plex Mono;'>RESEARCH BY:</div>
        <div style='font-size: 12px; font-weight: 500; color: #38BDF8 !important; font-family: IBM Plex Mono;'>AHMED HISHAM</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("LOAD WORKFILE (CSV/XLSX)", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("SIGNIFICANCE (α)", [0.01, 0.05, 0.10], index=1)
    
    if st.session_state['initialized']:
        st.markdown("<div style='font-size: 11px; color: #10B981;'>INTERFACE: ACTIVE</div>", unsafe_allow_html=True)
        if st.button("CLOSE WORKFILE", use_container_width=True):
            st.session_state['initialized'] = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# 04: WORKSPACE LOGIC
# ─────────────────────────────────────────────────────────────────
if uploaded_file and not st.session_state['initialized']:
    try:
        df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.markdown("### Workfile Initialization")
        c1, c2 = st.columns([2, 1])
        with c1:
            time_idx = st.selectbox("Temporal Anchor (Date/Observation)", options=df_raw.columns)
            df = df_raw.dropna(subset=[time_idx]).sort_values(by=time_idx)
            
            # Numeric Force & Date String Fix
            if df[time_idx].dtype in ['float64', 'int64']:
                df['display_date'] = df[time_idx].astype(int).astype(str)
            else:
                df['display_date'] = df[time_idx].astype(str)

            potential_vars = [c for c in df.columns if c not in [time_idx, 'display_date']]
            dep_var = st.selectbox("Dependent Variable (Y)", options=potential_vars)
            indep_vars = st.multiselect("Independent Variables (X)", options=[v for v in potential_vars if v != dep_var])
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if time_idx and dep_var and indep_vars:
                st.markdown(f"""<div class="nexus-card"><div class="label-mono">Capacity</div><div style="font-size:20px; font-weight:600;">{len(df)} Points</div></div>""", unsafe_allow_html=True)
                if st.button("OPEN INTERFACE", use_container_width=True):
                    for v in [dep_var] + indep_vars:
                        df[v] = pd.to_numeric(df[v], errors='coerce')
                    df = df.dropna(subset=[dep_var] + indep_vars)
                    st.session_state.update({"initialized": True, "df": df, "time": time_idx, "y": dep_var, "x": indep_vars})
                    st.rerun()
    except Exception as e:
        st.error(f"Kernel Error: {e}")

elif st.session_state['initialized']:
    df, time_idx, dep_var, indep_vars = st.session_state['df'], st.session_state['time'], st.session_state['y'], st.session_state['x']
    
    st.markdown(f"### Workspace: {dep_var} Analysis")
    tabs = st.tabs(["[ SUMMARY ]", "[ VISUALS ]", "[ STATIONARITY ]", "[ ESTIMATION ]", "[ DIAGNOSTICS ]"])

    with tabs[0]:
        st.markdown("<div class='label-mono'>[ DESCRIPTIVE STATISTICS ]</div>", unsafe_allow_html=True)
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
        fig = px.line(plot_df, x='display_date', y=[dep_var] + indep_vars, template="plotly_white")
        fig.update_layout(font_family="Inter", hovermode="x unified", xaxis_title=time_idx, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("<div class='label-mono'>[ UNIT ROOT TEST SPECIFICATION ]</div>", unsafe_allow_html=True)
        col_test, col_reg, col_lag = st.columns(3)
        with col_test: t_order = st.radio("Test at:", ["Level", "1st Difference", "2nd Difference"], horizontal=True)
        with col_reg: reg_t = st.selectbox("Include in Test:", ["c", "ct", "n"], format_func=lambda x: "Intercept" if x=="c" else "Trend" if x=="ct" else "None")
        with col_lag: lag_c = st.selectbox("Lag Selection", ["AIC", "BIC", "Fixed (1)"])

        for var in [dep_var] + indep_vars:
            series = df[var].dropna()
            label = var
            if t_order == "1st Difference": 
                series = series.diff().dropna(); label = f"Δ({var})"
            elif t_order == "2nd Difference": 
                series = series.diff().diff().dropna(); label = f"ΔΔ({var})"
            
            res = adfuller(series, regression=reg_t, autolag=(None if "Fixed" in lag_c else lag_c), maxlag=(1 if "Fixed" in lag_c else None))
            color = "#10B981" if res[1] < sig_level else "#F59E0B"
            st.markdown(f"<div style='border-left: 4px solid {color}; padding: 12px; background: white; border: 1px solid #E2E8F0; margin-bottom:8px;'><b>{label}</b>: {'STATIONARY' if res[1] < sig_level else 'NON-STATIONARY'}<br><span style='font-size:11px; color:gray;'>p-val: {res[1]:.4f} | Stat: {res[0]:.4f}</span></div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='label-mono'>[ ESTIMATION ENGINE ]</div>", unsafe_allow_html=True)
        method = st.radio("Technique", ["OLS", "ARDL", "VAR"], horizontal=True)
        
        if method == "OLS":
            use_const = st.checkbox("Include Constant (β₀)", value=True)
            st.latex(f"{dep_var}_t = {'\\beta_0 +' if use_const else ''} " + " + ".join([f"\\beta_{i+1} \\text{{{v}}}_t" for i, v in enumerate(indep_vars)]) + " + \\epsilon_t")
            if st.button("ESTIMATE OLS"):
                X = df[indep_vars]
                if use_const: X = add_constant(X)
                st.session_state['model_res'] = sm.OLS(df[dep_var], X).fit()

        elif method == "ARDL":
            max_l = st.slider("Max Lag", 1, 4, 1)
            if st.button("ESTIMATE ARDL"):
                sel = ardl_select_order(df[dep_var], max_l, df[indep_vars], max_l, ic='aic', trend='c')
                st.session_state['model_res'] = ARDL(df[dep_var], sel.model_order['dep'], df[indep_vars], sel.model_order['exog'], trend='c').fit()

        elif method == "VAR":
            v_lag = st.number_input("VAR Lag", 1, 3, 1)
            if st.button("ESTIMATE VAR"):
                from statsmodels.tsa.vector_ar.var_model import VAR
                st.session_state['model_res'] = VAR(df[[dep_var] + indep_vars]).fit(v_lag)

        if 'model_res' in st.session_state:
            res = st.session_state['model_res']
            st.markdown("<div class='label-mono'>[ OUTPUT ]</div>", unsafe_allow_html=True)
            if hasattr(res, 'params') and method != "VAR":
                out_df = pd.DataFrame({"Coef.": res.params, "Std.Err": res.bse, "t-Stat": res.tvalues, "Prob.": res.pvalues})
                st.table(out_df.style.format("{:.4f}"))
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**R-squared:** {res.rsquared:.6f}")
                    st.write(f"**Adj R-squared:** {res.rsquared_adj:.6f}")
                    st.write(f"**F-statistic:** {res.fvalue:.4f}")
                with c2:
                    st.write(f"**AIC:** {res.aic:.4f}")
                    st.write(f"**BIC:** {res.bic:.4f}")
                    st.write(f"**Durbin-Watson:** {durbin_watson(res.resid):.2f}")
            elif method == "VAR":
                st.text(res.summary())

    with tabs[4]:
        if 'model_res' in st.session_state and method != "VAR":
            res = st.session_state['model_res']
            st.markdown("<div class='label-mono'>[ DIAGNOSTICS ]</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(x=df['display_date'], y=res.resid, title="Residuals", template="plotly_white").add_hline(y=0, line_dash="dash", line_color="red"), use_container_width=True)
            with c2:
                jb_stat, jb_p, _, _ = jarque_bera(res.resid)
                st.plotly_chart(px.histogram(res.resid, title="Normality Check", template="plotly_white"), use_container_width=True)
                st.write(f"**Jarque-Bera p-val:** {jb_p:.4f}")
else:
    st.markdown("<div style='text-align:center; padding: 100px; color:#94A3B8; font-family:IBM Plex Mono;'>LOAD DATASET TO INITIALIZE INTERFACE</div>", unsafe_allow_html=True)
