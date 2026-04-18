import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

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

    .data-card {
        background: white;
        border: 1px solid var(--nexus-border);
        padding: 1.5rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; color: #94A3B8; margin-bottom: 8px; }
    .card-value { font-size: 24px; font-weight: 600; color: #0F172A; }

    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }
    
    /* Analysis Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
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
    sig_level = st.selectbox("SIGNIFICANCE LEVEL (α)", [0.01, 0.05, 0.10], index=1)
    st.markdown("<div style='font-size:10px; opacity:0.6;'>KERNEL v2.5 • RESEARCH MODE</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    # Aggressive Reader logic to ensure numeric integrity
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # Force numeric conversion for analysis columns
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # --- STEP 1 & 2: INTEGRITY HUB & ANATOMY WIZARD ---
    st.markdown("<div class='section-header'>Step 1 & 2: Data Integrity & Mapping</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])

    with col_map:
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df_raw.columns)
        potential_vars = [c for c in df_raw.columns if c != time_idx]
        
        # Clean based on Year availability
        df = df_raw.dropna(subset=[time_idx]).copy()
        df = df[df[time_idx] > 1900].sort_values(by=time_idx)
        
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        is_mapped = time_idx and dep_var and indep_vars
        if is_mapped:
            st.markdown(f"""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>Range: {int(df[time_idx].min())}-{int(df[time_idx].max())}<br>Obs: {len(df)} Years</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("INITIALIZE ANALYSIS KERNEL"):
                st.session_state['initialized'] = True
        else:
            st.markdown("""
            <div style='background: #FEFCE8; border: 1px solid #FEF08A; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #854D0E; font-size: 11px; font-weight: 600; font-family: IBM Plex Mono;'>[ PENDING ]</div>
                <div style='color: #854D0E; font-size: 12px; margin-top:5px;'>Awaiting variable mapping.</div>
            </div>
            """, unsafe_allow_html=True)

    # --- STEP 3: THE EXPLORATORY KERNEL (Visuals & Health) ---
    if st.session_state.get('initialized', False):
        st.markdown("<div class='section-header'>Step 3: Exploratory Intelligence</div>", unsafe_allow_html=True)
        
        # Professional Tabs for Visuals vs Science
        tab_viz, tab_health = st.tabs(["[ VISUAL CONVERGENCE ]", "[ STATIONARITY AUDIT ]"])
        
        with tab_viz:
            # Multi-variable time series plot
            plot_vars = [dep_var] + indep_vars
            fig = px.line(df, x=time_idx, y=plot_vars, 
                          template="plotly_white",
                          color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B", "#EF4444"])
            
            fig.update_layout(
                font_family="Inter",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="",
                yaxis_title="Observed Units"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_health:
            st.markdown("#### Augmented Dickey-Fuller (ADF) Summary")
            st.write("Verifying stationarity to prevent spurious regression.")
            
            for var in [dep_var] + indep_vars:
                series = df[var].dropna()
                # Run the math
                adf_res = adfuller(series)
                pval = adf_res[1]
                is_stationary = pval < sig_level
                
                # Professional Verdict Design
                border_color = "#10B981" if is_stationary else "#F59E0B"
                verdict = "STATIONARY (I(0))" if is_stationary else "NON-STATIONARY (Unit Root)"
                
                st.markdown(f"""
                <div style='border: 1px solid #E2E8F0; padding: 1.2rem; border-left: 5px solid {border_color}; margin-bottom: 12px; background: white;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-family: IBM Plex Mono; font-size: 14px; font-weight: 600;'>{var}</span>
                        <span style='font-family: IBM Plex Mono; font-size: 11px; color: {border_color}; font-weight: 600;'>{verdict}</span>
                    </div>
                    <div style='font-size: 12px; color: #64748B; margin-top: 8px;'>
                        Test Statistic: {adf_res[0]:.4f} | p-value: {pval:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if not all(adfuller(df[v].dropna())[1] < sig_level for v in [dep_var] + indep_vars):
                st.warning("⚠️ DETECTION: Mixed levels of integration. Consider ARDL model or first-differencing for I(1) variables.")

else:
    st.markdown("""
    <div style='border: 1px dashed #E2E8F0; padding: 5rem; text-align: center; color: #94A3B8; font-family: IBM Plex Mono; font-size: 12px;'>
        AWAITING DATASET UPLOAD TO INITIALIZE WORKSPACE...
    </div>
    """, unsafe_allow_html=True)
