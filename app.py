import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

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
    .section-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase;
        letter-spacing: 1.5px; color: #64748B; margin: 2rem 0 1rem 0;
        border-bottom: 1px solid var(--nexus-border); padding-bottom: 5px;
    }
    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

# Navigation Header
st.markdown("""
<div style='padding: 1.5rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 2rem;'>
    <div style='font-size: 20px; font-weight: 700; color: #0F172A;'>NEXUS <span style='font-weight:300; color:#64748B;'>ECONOMETRICS</span></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><div style='font-family:IBM Plex Mono; font-size:11px;'>[ 01 ] INGESTION</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("CONFIDENCE (α)", [0.01, 0.05, 0.10], index=1)
    if st.session_state['initialized']:
        if st.button("RESET KERNEL", use_container_width=True):
            st.session_state['initialized'] = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# DATA PROCESSING KERNEL
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    try:
        # 1. Flexible Loader
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.markdown("<div class='section-header'>Step 1 & 2: Structural Mapping</div>", unsafe_allow_html=True)
        c_map, c_status = st.columns([2, 1])

        with c_map:
            time_idx = st.selectbox("Select Temporal Axis (Year)", options=df_raw.columns)
            
            # Smart Variable Detection (Robust Numeric Check)
            numeric_cols = []
            for col in df_raw.columns:
                if col != time_idx:
                    # Try to clean strings like "1,200" or "$50"
                    series_clean = df_raw[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                    temp_num = pd.to_numeric(series_clean, errors='coerce')
                    if temp_num.notna().sum() > 0:
                        numeric_cols.append(col)
            
            dep_var = st.selectbox("Assign Target Variable (Y)", options=numeric_cols)
            indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in numeric_cols if v != dep_var])

        with c_status:
            st.markdown("<br>", unsafe_allow_html=True)
            if time_idx and dep_var and len(indep_vars) > 0:
                # 2. Strict Convergence Cleaning
                analysis_cols = [time_idx, dep_var] + indep_vars
                df = df_raw[analysis_cols].copy()
                
                # Robust Conversion
                for col in analysis_cols:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows where ANY of the selected variables are missing
                df = df.dropna().sort_values(by=time_idx)
                
                if not df.empty:
                    st.markdown(f"""
                    <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                        <div style='color: #166534; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                        <div style='color: #166534; font-size: 12px; margin-top:5px;'>
                            Converged Sample: {len(df)} Years<br>
                            Span: {int(df[time_idx].min())} - {int(df[time_idx].max())}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("INITIALIZE KERNEL", use_container_width=True):
                        st.session_state['initialized'] = True
                        st.rerun()
                else:
                    # DEBUGGER VIEW: Tell the user WHY it's empty
                    st.error("⚠️ DATA CONVERGENCE FAILED")
                    st.write("The selected variables do not share any common years. Check counts below:")
                    # Show a small table of available data points per variable
                    avail = df_raw[analysis_cols].notna().sum()
                    st.dataframe(avail.rename("Available Points"))
            else:
                st.markdown("<div style='background: #FEFCE8; border: 1px solid #FEF08A; padding: 1.5rem; border-radius: 4px; color: #854D0E; font-size: 12px;'>Map all variables to begin.</div>", unsafe_allow_html=True)

        # 3. Step 3: Analysis Tabs
        if st.session_state['initialized']:
            st.markdown("<div class='section-header'>Step 3: Exploratory Intelligence</div>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["[ VISUAL PULSE ]", "[ STATIONARITY AUDIT ]"])
            
            with t1:
                fig = px.line(df, x=time_idx, y=[dep_var] + indep_vars, template="plotly_white")
                fig.update_layout(font_family="Inter", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.markdown("#### ADF Stationarity Analysis")
                for var in [dep_var] + indep_vars:
                    res = adfuller(df[var].dropna())
                    is_stat = res[1] < sig_level
                    color = "#10B981" if is_stat else "#F59E0B"
                    st.markdown(f"""
                    <div style='border-left: 5px solid {color}; padding: 1rem; background: white; margin-bottom: 10px; border: 1px solid #E2E8F0;'>
                        <strong>{var}</strong>: {"STATIONARY" if is_stat else "NON-STATIONARY"}<br>
                        <span style='font-size: 11px; color: gray;'>p-value: {res[1]:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Kernel Initialization Error: {e}")
else:
    st.markdown("<div style='text-align:center; padding: 5rem; color: #94A3B8; font-family: IBM Plex Mono;'>AWAITING DATASET...</div>", unsafe_allow_html=True)
