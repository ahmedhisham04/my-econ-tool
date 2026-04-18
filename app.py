import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# NEXUS PROFESSIONAL DESIGN SYSTEM (UI/UX Foundation)
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
# SIDEBAR: INGESTION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><div style='font-family:IBM Plex Mono; font-size:11px;'>[ 01 ] INGESTION</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD RESEARCH DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    st.markdown("<div style='font-size:10px; opacity:0.6;'>KERNEL v2.4 • RESEARCH MODE</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN WORKSPACE
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    # --- STEP 1: THE INTELLIGENCE HUB ---
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # GHOST KILLER: Remove purely empty rows/columns
    df = df_raw.dropna(how='all').dropna(axis=1, how='all')

    st.markdown("<div class='section-header'>Step 1: Data Integrity Profile</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="data-card"><div class="card-label">Sample Size</div><div class="card-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with c2:
        num_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown(f'<div class="data-card"><div class="card-label">Numeric Columns</div><div class="card-value">{len(num_vars)}</div></div>', unsafe_allow_html=True)
    with c3:
        gaps = df.isnull().sum().sum()
        st.markdown(f'<div class="data-card"><div class="card-label">Detected Gaps</div><div class="card-value">{gaps}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="data-card"><div class="card-label">Integrity</div><div class="card-value">{"High" if gaps == 0 else "Review"}</div></div>', unsafe_allow_html=True)

    # --- STEP 2: THE ANATOMY WIZARD ---
    st.markdown("<div class='section-header'>Step 2: Data Anatomy Wizard</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])

    with col_map:
        # User identifies the Time index
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df.columns)
        
        # We strictly exclude the Time Index from being an analysis variable
        potential_vars = [c for c in df.columns if c != time_idx]
        
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if time_idx and dep_var and indep_vars:
            st.markdown("""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 12px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>Variables mapped. Research kernel unlocked.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("INITIALIZE ANALYSIS"):
                st.success("Kernel Initialized. Proceed to Step 3.")
        else:
            st.markdown("""
            <div style='background: #FEFCE8; border: 1px solid #FEF08A; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #854D0E; font-size: 12px; font-weight: 600; font-family: IBM Plex Mono;'>[ PENDING ]</div>
                <div style='color: #854D0E; font-size: 12px; margin-top:5px;'>Awaiting variable role assignment.</div>
            </div>
            """, unsafe_allow_html=True)

    # Preview of the mapped sample
    with st.expander("VIEW CLEANED MAPPING (TOP 10)"):
        st.dataframe(df[[time_idx, dep_var] + indep_vars].head(10), use_container_width=True)

else:
    st.markdown("""
    <div style='border: 1px dashed #E2E8F0; padding: 5rem; text-align: center; color: #94A3B8; font-family: IBM Plex Mono; font-size: 12px;'>
        AWAITING DATASET UPLOAD...
    </div>
    """, unsafe_allow_html=True)
