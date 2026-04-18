import streamlit as st
import pandas as pd
import numpy as np

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

    /* Clean Sidebar */
    [data-testid="stSidebar"] { background-color: var(--nexus-sidebar) !important; border-right: 1px solid var(--nexus-border); }
    [data-testid="stSidebar"] * { color: #94A3B8 !important; }

    /* The "First Sight" Header */
    .hero-container {
        padding: 2rem 0;
        border-bottom: 1px solid var(--nexus-border);
        margin-bottom: 2rem;
    }
    .hero-title { font-size: 28px; font-weight: 600; color: #0F172A; letter-spacing: -0.5px; }
    .hero-subtitle { font-size: 14px; color: #64748B; margin-top: 5px; }

    /* Professional Variable Cards */
    .data-card {
        background: white;
        border: 1px solid var(--nexus-border);
        padding: 1.5rem;
        border-radius: 4px;
        transition: transform 0.2s ease;
    }
    .data-card:hover { border-color: var(--nexus-accent); }
    .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; color: #94A3B8; }
    .card-value { font-size: 22px; font-weight: 600; color: #0F172A; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# COMPONENT 1: THE INTELLIGENCE HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Nexus Econometrics</div>
    <div class="hero-subtitle">Advanced Research Environment • Data Integrity Engine Active</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# COMPONENT 2: THE INGESTION SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding: 1rem 0;'>[ NAVIGATION ]</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("LOAD RESEARCH DATASET", type=["csv", "xlsx"])
    st.markdown("---")
    st.markdown("<div style='font-size: 11px;'>SYSTEM STATUS: OPERATIONAL</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# COMPONENT 3: THE "FIRST SIGHT" DATA DASHBOARD
# ─────────────────────────────────────────────────────────────────
if uploaded_file:
    # Logic: Read and immediately purge ghost rows
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # THE GHOST KILLER: Drops rows that are purely empty
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # DETECT TEMPORAL AXIS: Finding the "Year" column
    potential_time = [c for c in df.columns if any(k in c.lower() for k in ['year', 'date', 'time', 'period'])]
    
    # RENDER THE HUB
    st.markdown("### DATA INTEGRITY PROFILE")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="data-card"><div class="card-label">Sample Size</div><div class="card-value">{len(df)}</div></div>""", unsafe_allow_html=True)
    with c2:
        # Strict logic: Find only truly numeric data columns
        num_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        st.markdown(f"""<div class="data-card"><div class="card-label">Variables</div><div class="card-value">{len(num_vars)}</div></div>""", unsafe_allow_html=True)
    with c3:
        missing_cells = df.isnull().sum().sum()
        st.markdown(f"""<div class="data-card"><div class="card-label">Data Gaps</div><div class="card-value">{missing_cells}</div></div>""", unsafe_allow_html=True)
    with c4:
        freq_detected = "Annual" if len(df) < 100 else "High-Freq"
        st.markdown(f"""<div class="data-card"><div class="card-label">Est. Frequency</div><div class="card-value">{freq_detected}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### THE VARIABLE UNIVERSE")
    st.write("Review your detected variables below before proceeding to analysis.")
    st.dataframe(df.head(10), use_container_width=True)

else:
    # The "Invitation" State
    st.markdown("""
    <div style='background: white; border: 1px dashed #CBD5E1; padding: 4rem; text-align: center; border-radius: 4px;'>
        <div style='color: #64748B; font-size: 14px; font-family: "IBM Plex Mono", monospace;'>
            AWAITING DATA INPUT...<br>
            DRAG AND DROP CSV OR XLSX TO INITIALIZE WORKBENCH
        </div>
    </div>
    """, unsafe_allow_html=True)
