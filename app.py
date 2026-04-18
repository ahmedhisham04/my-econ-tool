import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────────────────────────
# NEXUS DESIGN SYSTEM (Professional, Neutral, Research-Grade)
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

    /* Custom Section Headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748B;
        margin: 2rem 0 1rem 0;
        border-bottom: 1px solid var(--nexus-border);
        padding-bottom: 5px;
    }

    /* Professional Variable Cards */
    .data-card {
        background: white;
        border: 1px solid var(--nexus-border);
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; margin-bottom: 8px; }
    .card-value { font-size: 24px; font-weight: 600; color: #0F172A; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TOP NAVIGATION
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1.5rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 2rem;'>
    <div style='font-size: 20px; font-weight: 700; color: #0F172A;'>NEXUS <span style='font-weight:300; color:#64748B;'>ECONOMETRICS</span></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR: INGESTION (Step 1)
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><div style='font-family:IBM Plex Mono; font-size:11px;'>[ 01 ] INGESTION</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD RESEARCH DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    sig_level = st.selectbox("CONFIDENCE LEVEL (α)", [0.01, 0.05, 0.10], index=1)
    st.markdown("<div style='font-size:10px; opacity:0.6;'>KERNEL v2.5 • RESEARCH MODE</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    # --- AGGRESSIVE INGESTION ---
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # Force all columns to numeric to capture GDP/Inflation metadata errors
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # --- STEP 1 & 2: INTEGRITY HUB & ANATOMY WIZARD ---
    st.markdown("<div class='section-header'>Step 1 & 2: Integrity & Mapping</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])

    with col_map:
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df_raw.columns)
        potential_vars = [c for c in df_raw.columns if c != time_idx]
        
        # CLEANING DATA BASED ON YEAR (The Ghost Killer)
        df = df_raw.dropna(subset=[time_idx]).copy()
        df = df[df[time_idx] > 1900] # Ensure we only keep valid year observations
        df = df.sort_values(by=time_idx)
        
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if time_idx and dep_var and indep_vars:
            st.markdown(f"""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>
                    Sample: {int(df[time_idx].min())} - {int(df[time_idx].max())}<br>
                    Observations: {len(df)} Years
                </div>
            </div>
            """, unsafe_allow_html=True)
            init_btn = st.button("INITIALIZE RESEARCH KERNEL")
        else:
            init_btn = False
            st.markdown("""
            <div style='background: #FEFCE8; border: 1px solid #FEF08A; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #854D0E; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ PENDING ]</div>
                <div style='color: #854D0E; font-size: 12px; margin-top:5px;'>Awaiting variable mapping.</div>
            </div>
            """, unsafe_allow_html=True)

    # --- STEP 3: THE ANALYSIS KERNEL ---
    if init_btn or st.session_state.get('initialized', False):
        st.session_state['initialized'] = True
        st.markdown("<div class='section-header'>Step 3: Analysis Kernel (Visuals & Audit)</div>", unsafe_allow_html=True)
        
        viz_tab, audit_tab = st.tabs(["[ VISUAL CONVERGENCE ]", "[ STATIONARITY AUDIT ]"])
        
        with viz_tab:
            plot_vars = [dep_var] + indep_vars
            fig = px.line(df, x=time_idx, y=plot_vars, 
                          template="plotly_white",
                          color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"])
            
            fig.update_layout(
                font_family="Inter",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="",
                yaxis_title="Observed Value"
            )
            st.plotly_chart(fig, use_container_width=True)

        with audit_tab:
            st.markdown("#### Stationarity Assessment (Augmented Dickey-Fuller)")
            for var in [dep_var] + indep_vars:
                series = df[var].dropna()
                res = adfuller(series)
                is_stationary = res[1] < sig_level
                
                color = "#10B981" if is_stationary else "#F59E0B"
                status = "STATIONARY (I(0))" if is_stationary else "NON-STATIONARY (Trend)"
                
                st.markdown(f"""
                <div style='border: 1px solid #E2E8F0; padding: 1rem; border-left: 4px solid {color}; margin-bottom: 10px; background: white;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-family: IBM Plex Mono; font-size: 13px; font-weight: 600;'>{var}</span>
                        <span style='font-family: IBM Plex Mono; font-size: 11px; color: {color};'>{status}</span>
                    </div>
                    <div style='font-size: 11px; color: #64748B; margin-top: 5px;'>
                        ADF Stat: {res[0]:.4f} | p-value: {res[1]:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='border: 1px dashed #E2E8F0; padding: 5rem; text-align: center; color: #94A3B8; font-family: IBM Plex Mono; font-size: 12px;'>
        AWAITING DATASET UPLOAD...
    </div>
    """, unsafe_allow_html=True)
