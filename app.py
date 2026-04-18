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
    .data-card {
        background: white; border: 1px solid var(--nexus-border);
        padding: 1.5rem; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; margin-bottom: 8px; }
    .card-value { font-size: 24px; font-weight: 600; color: #0F172A; }
    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

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
    uploaded_file = st.file_uploader("UPLOAD RESEARCH DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("CONFIDENCE LEVEL (α)", [0.01, 0.05, 0.10], index=1)

# ─────────────────────────────────────────────────────────────────
# WORKSPACE
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    # Aggressive Load
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # Force Numeric
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    st.markdown("<div class='section-header'>Step 1 & 2: Anatomy & Mapping</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])

    with col_map:
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df_raw.columns)
        potential_vars = [c for c in df_raw.columns if c != time_idx]
        
        # Clean Data
        df = df_raw.dropna(subset=[time_idx]).copy()
        df = df[df[time_idx] > 1900].sort_values(by=time_idx)
        
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if time_idx and dep_var and indep_vars and not df.empty:
            # SAFE RANGE CALCULATION (Fixes the ValueError)
            t_min = int(df[time_idx].min())
            t_max = int(df[time_idx].max())
            
            st.markdown(f"""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>
                    Range: {t_min} - {t_max}<br>
                    Obs: {len(df)} Years
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("INITIALIZE RESEARCH KERNEL"):
                st.session_state['initialized'] = True
        else:
            st.markdown("""
            <div style='background: #FEFCE8; border: 1px solid #FEF08A; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #854D0E; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ PENDING ]</div>
                <div style='color: #854D0E; font-size: 12px; margin-top:5px;'>Awaiting valid variable mapping.</div>
            </div>
            """, unsafe_allow_html=True)

    # --- STEP 3: THE KERNEL ---
    if st.session_state.get('initialized', False):
        st.markdown("<div class='section-header'>Step 3: Exploratory Intelligence</div>", unsafe_allow_html=True)
        t_viz, t_audit = st.tabs(["[ VISUAL CONVERGENCE ]", "[ STATIONARITY AUDIT ]"])
        
        with t_viz:
            plot_vars = [dep_var] + indep_vars
            fig = px.line(df, x=time_idx, y=plot_vars, template="plotly_white")
            fig.update_layout(font_family="Inter", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with t_audit:
            st.markdown("#### Stationarity Assessment (ADF)")
            for var in [dep_var] + indep_vars:
                res = adfuller(df[var].dropna())
                is_stat = res[1] < sig_level
                color = "#10B981" if is_stat else "#F59E0B"
                st.markdown(f"""
                <div style='border-left: 5px solid {color}; padding: 1rem; background: white; margin-bottom: 10px; border: 1px solid #E2E8F0; border-left: 5px solid {color};'>
                    <strong>{var}</strong>: {"STATIONARY" if is_stat else "NON-STATIONARY"}<br>
                    <span style='font-size: 11px; color: gray;'>p-value: {res[1]:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("<div style='text-align:center; padding: 5rem; color: #94A3B8;'>AWAITING DATASET...</div>", unsafe_allow_html=True)
