"""
Time Series Econometric Analysis Suite
A professional Streamlit application for applied econometrics.
Author: Generated for LinkedIn-ready deployment.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.tsa.stattools import kpss

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EconoMetrics Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Dark navy background */
  .stApp {
    background-color: #0b0f1a;
    color: #e0e6f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%);
    border-right: 1px solid #1e2d4a;
  }

  section[data-testid="stSidebar"] * {
    color: #c8d8f0 !important;
  }

  /* Logo / header in sidebar */
  .sidebar-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #4fc3f7 !important;
    letter-spacing: 0.05em;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 1rem;
  }

  .sidebar-logo span {
    color: #ffd54f !important;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0d1e35 0%, #112244 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
  }

  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #4fc3f7, #0288d1);
    border-radius: 10px 0 0 10px;
  }

  .metric-card.gold::before {
    background: linear-gradient(180deg, #ffd54f, #f9a825);
  }

  .metric-card.green::before {
    background: linear-gradient(180deg, #69f0ae, #00c853);
  }

  .metric-card.red::before {
    background: linear-gradient(180deg, #ff5252, #b71c1c);
  }

  .metric-label {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #7ba7d0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
  }

  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f4ff;
    font-family: 'IBM Plex Mono', monospace;
  }

  .metric-sub {
    font-size: 0.75rem;
    color: #5a7fa0;
    margin-top: 0.2rem;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border-bottom: 1px solid #1e3a5f;
    gap: 0;
  }

  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a7fa0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    padding: 0.8rem 1.4rem;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }

  .stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
    background: transparent !important;
  }

  .stTabs [data-baseweb="tab-panel"] {
    background: transparent;
    padding-top: 1.5rem;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0288d1, #01579b);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s;
    width: 100%;
  }

  .stButton > button:hover {
    background: linear-gradient(135deg, #4fc3f7, #0288d1);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
  }

  /* Download button */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #00695c, #004d40) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    width: 100%;
  }

  /* Select boxes & inputs */
  .stSelectbox > div > div {
    background: #0d1e35;
    border: 1px solid #1e3a5f;
    color: #c8d8f0;
    border-radius: 6px;
  }

  /* Info / success / warning boxes */
  .stAlert {
    border-radius: 8px;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4fc3f7;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
  }

  /* LaTeX display */
  .latex-box {
    background: #060d1a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    text-align: center;
  }

  /* Results table */
  .stDataFrame {
    background: #0d1e35 !important;
  }

  /* Divider */
  hr {
    border-color: #1e3a5f;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0b0f1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #2a4d70; }

  /* Number input */
  .stNumberInput input {
    background: #0d1e35;
    border: 1px solid #1e3a5f;
    color: #c8d8f0;
    border-radius: 6px;
  }

  /* Multiselect */
  .stMultiSelect > div > div {
    background: #0d1e35;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
  }

  /* Radio */
  .stRadio label { color: #c8d8f0 !important; }

  /* Expander */
  .streamlit-expanderHeader {
    background: #0d1e35 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #4fc3f7 !important;
  }

  /* Header banner */
  .app-header {
    background: linear-gradient(135deg, #0d1e35 0%, #081428 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8f4ff;
    letter-spacing: 0.02em;
  }

  .app-title span { color: #4fc3f7; }

  .app-subtitle {
    font-size: 0.85rem;
    color: #5a7fa0;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .badge {
    background: linear-gradient(135deg, #0288d1, #01579b);
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
  }

  /* Interpret badge */
  .interpret-stationary {
    background: rgba(105, 240, 174, 0.12);
    border: 1px solid #00c853;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #69f0ae;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
  }

  .interpret-nonstationary {
    background: rgba(255, 213, 79, 0.08);
    border: 1px solid #f9a825;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #ffd54f;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Plotly template ───────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#0d1526",
        font=dict(family="IBM Plex Mono, monospace", color="#c8d8f0", size=12),
        xaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        colorway=["#4fc3f7", "#ffd54f", "#69f0ae", "#ff8a65", "#ce93d8", "#80cbc4"],
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_data(uploaded_file):
    """Load CSV or Excel file, returning a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df


def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def run_adf(series: pd.Series, regression="c"):
    """Return ADF results dict."""
    clean = series.dropna()
    result = adfuller(clean, regression=regression, autolag="AIC")
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Obs Used": result[3],
        "Critical Values": result[4],
    }


def run_kpss(series: pd.Series, regression="c"):
    clean = series.dropna()
    stat, pval, lags, crit = kpss(clean, regression=regression, nlags="auto")
    return {"KPSS Stat": stat, "p-value": pval, "Lags": lags, "Critical Values": crit}


def metric_html(label, value, sub=None, color="blue"):
    color_cls = {"blue": "", "gold": "gold", "green": "green", "red": "red"}.get(color, "")
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card {color_cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {sub_html}
    </div>
    """


def plotly_ts(df, cols, title="Time Series"):
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(width=2), mode="lines"
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=title, font=dict(size=14, color="#c8d8f0")),
        hovermode="x unified",
        legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1),
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plotly_corr(df, cols):
    sub = df[cols].dropna()
    corr = sub.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="Correlation Heatmap", font=dict(size=14, color="#c8d8f0")),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar=dict(tickfont=dict(color="#c8d8f0")),
    )
    return fig


def acf_pacf_plot(series, nlags=24):
    clean = series.dropna()
    acf_vals = acf(clean, nlags=nlags, fft=True)
    pacf_vals = pacf(clean, nlags=nlags)
    lags = list(range(len(acf_vals)))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    ci = 1.96 / np.sqrt(len(clean))

    for vals, col in [(acf_vals, 1), (pacf_vals, 2)]:
        for i, v in enumerate(vals):
            fig.add_trace(go.Bar(x=[i], y=[v], marker_color="#4fc3f7", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=lags, y=[ci]*len(lags), mode="lines",
                                  line=dict(dash="dash", color="#ffd54f", width=1), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=lags, y=[-ci]*len(lags), mode="lines",
                                  line=dict(dash="dash", color="#ffd54f", width=1), showlegend=False), row=1, col=col)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#c8d8f0"
    return fig


def residual_diagnostics_plot(resid):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Residuals Over Time", "Residual Distribution",
                        "ACF of Residuals", "Residuals vs Fitted (Q-Q)"],
    )
    t = list(range(len(resid)))
    ci = 1.96 / np.sqrt(len(resid))
    acf_r = acf(resid, nlags=20, fft=True)

    # Residuals over time
    fig.add_trace(go.Scatter(x=t, y=resid, mode="lines",
                              line=dict(color="#4fc3f7", width=1.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[0]*len(t), mode="lines",
                              line=dict(color="#ffd54f", dash="dash", width=1), showlegend=False), row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(x=resid, nbinsx=30, marker_color="#4fc3f7",
                                opacity=0.7, showlegend=False), row=1, col=2)

    # ACF of residuals
    for i, v in enumerate(acf_r):
        fig.add_trace(go.Bar(x=[i], y=[v], marker_color="#69f0ae", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(acf_r))), y=[ci]*len(acf_r), mode="lines",
                              line=dict(dash="dash", color="#ffd54f", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(acf_r))), y=[-ci]*len(acf_r), mode="lines",
                              line=dict(dash="dash", color="#ffd54f", width=1), showlegend=False), row=2, col=1)

    # Q-Q plot
    from scipy import stats as scipy_stats
    qq = scipy_stats.probplot(resid)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                              marker=dict(color="#4fc3f7", size=5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*np.array(qq[0][0]),
                              mode="lines", line=dict(color="#ffd54f", width=1.5), showlegend=False), row=2, col=2)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#c8d8f0"
    return fig


def results_to_csv(results_dict):
    rows = []
    for k, v in results_dict.items():
        rows.append({"Metric": k, "Value": v})
    return pd.DataFrame(rows).to_csv(index=False).encode()


def results_to_txt(model_type, summary_str, adf_results=None):
    lines = [
        "=" * 70,
        f"  EconoMetrics Suite — {model_type} Results",
        "=" * 70, "",
    ]
    if adf_results:
        lines += ["[ ADF Test ]", str(adf_results), ""]
    lines += [summary_str]
    return "\n".join(lines).encode()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">Econo<span>Metrics</span><br>Suite v1.0</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="section-header">📂 Data Import</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload a time series dataset. Numeric columns are auto-detected.",
    )

    st.markdown('<p class="section-header">⚙️ Options</p>', unsafe_allow_html=True)
    sig_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], index=1,
                              format_func=lambda x: f"{int(x*100)}%")
    adf_regression = st.selectbox("ADF Regression Type",
                                   ["c", "ct", "ctt", "nc"],
                                   format_func=lambda x: {
                                       "c": "Constant only",
                                       "ct": "Constant + Trend",
                                       "ctt": "Constant + Trend²",
                                       "nc": "No constant",
                                   }[x])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#3a5a7a; font-family: IBM Plex Mono, monospace;'>
    Built with Streamlit · Statsmodels<br>
    Plotly · Pandas · NumPy<br><br>
    🇪🇬 Cairo, Egypt
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main Content
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">Econo<span>Metrics</span> Suite</div>
    <div class="app-subtitle">Time Series Econometric Analysis · ADF · OLS · ARDL · VAR</div>
  </div>
  <div class="badge">PROFESSIONAL EDITION</div>
</div>
""", unsafe_allow_html=True)


# ── No data state ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem;'>
      <div style='font-size:4rem; margin-bottom:1rem;'>📊</div>
      <div style='font-family: IBM Plex Mono, monospace; font-size:1.1rem; color:#4fc3f7; margin-bottom:0.8rem;'>
        Upload a dataset to begin
      </div>
      <div style='font-size:0.85rem; color:#3a5a7a; max-width:500px; margin:0 auto;'>
        Supports CSV and Excel formats. The app auto-detects numeric columns and guides you through
        stationarity testing, model selection, and diagnostic evaluation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Step 1</div>
          <div style='color:#4fc3f7; font-size:0.95rem; font-family: IBM Plex Mono; margin-top:0.4rem;'>
            Upload Data
          </div>
          <div class="metric-sub">CSV or Excel time series data</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card gold">
          <div class="metric-label">Step 2</div>
          <div style='color:#ffd54f; font-size:0.95rem; font-family: IBM Plex Mono; margin-top:0.4rem;'>
            Test Stationarity
          </div>
          <div class="metric-sub">ADF & KPSS tests with auto-interpretation</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card green">
          <div class="metric-label">Step 3</div>
          <div style='color:#69f0ae; font-size:0.95rem; font-family: IBM Plex Mono; margin-top:0.4rem;'>
            Build & Diagnose
          </div>
          <div class="metric-sub">OLS · ARDL · VAR + full diagnostics</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

numeric_cols = get_numeric_cols(df)
if len(numeric_cols) == 0:
    st.error("No numeric columns detected. Please check your file.")
    st.stop()

# Try to detect date index
date_col = None
for col in df.columns:
    if df[col].dtype == "object" or "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
        try:
            parsed = pd.to_datetime(df[col])
            df.index = parsed
            date_col = col
            df = df.drop(columns=[col])
            numeric_cols = get_numeric_cols(df)
            break
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Data Preview",
    "🔬  Stationarity Tests",
    "🏗️  Model Building",
    "🩺  Diagnostics",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Preview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_html("Observations", f"{len(df):,}", "rows in dataset"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html("Variables", str(len(numeric_cols)), "numeric columns", "gold"), unsafe_allow_html=True)
    with m3:
        missing_pct = df[numeric_cols].isna().mean().mean() * 100
        st.markdown(metric_html("Missing", f"{missing_pct:.1f}%", "average across cols",
                                "red" if missing_pct > 5 else "green"), unsafe_allow_html=True)
    with m4:
        if isinstance(df.index, pd.DatetimeIndex):
            freq = f"{df.index.min().strftime('%Y')}–{df.index.max().strftime('%Y')}"
        else:
            freq = f"Rows 0–{len(df)-1}"
        st.markdown(metric_html("Index Range", freq, "time coverage", "blue"), unsafe_allow_html=True)

    st.markdown("---")

    # Interactive time series chart
    st.markdown('<p class="section-header">Interactive Time Series</p>', unsafe_allow_html=True)
    selected_preview = st.multiselect("Select variables to plot", numeric_cols,
                                       default=numeric_cols[:min(3, len(numeric_cols))])
    if selected_preview:
        st.plotly_chart(plotly_ts(df, selected_preview, "Time Series Overview"),
                        use_container_width=True)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
        heatmap_cols = st.multiselect("Variables for heatmap", numeric_cols,
                                       default=numeric_cols[:min(6, len(numeric_cols))],
                                       key="hm")
        if len(heatmap_cols) >= 2:
            st.plotly_chart(plotly_corr(df, heatmap_cols), use_container_width=True)

    # Data table
    with st.expander("📄 Raw Data Table"):
        st.dataframe(df[numeric_cols].style.format("{:.4f}"), use_container_width=True)
        st.download_button("⬇ Download Dataset as CSV", df.to_csv().encode(),
                           file_name="dataset.csv", mime="text/csv")

    # Summary statistics
    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(df[numeric_cols].describe().T.style.format("{:.4f}"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Stationarity Tests
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Unit Root & Stationarity Testing</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        test_var = st.selectbox("Select Variable", numeric_cols)
        run_kpss_flag = st.checkbox("Also run KPSS test", value=True)
        diff_order = st.radio("Differencing", [0, 1, 2],
                               format_func=lambda x: ["Level I(0)", "1st Diff I(1)", "2nd Diff I(2)"][x])
        run_test = st.button("▶ Run Tests")

    with col_right:
        if run_test:
            series = df[test_var].dropna()
            if diff_order == 1:
                series = series.diff().dropna()
            elif diff_order == 2:
                series = series.diff().diff().dropna()

            label = f"{test_var}" + ["", " [Δ]", " [Δ²]"][diff_order]

            # Plot the series
            fig_ts = plotly_ts(pd.DataFrame({label: series}), [label],
                               f"Series: {label}")
            st.plotly_chart(fig_ts, use_container_width=True)

    if run_test:
        series = df[test_var].dropna()
        if diff_order == 1:
            series = series.diff().dropna()
        elif diff_order == 2:
            series = series.diff().diff().dropna()

        label = f"{test_var}" + ["", " [Δ]", " [Δ²]"][diff_order]

        # ADF
        st.markdown('<p class="section-header">Augmented Dickey-Fuller Test</p>', unsafe_allow_html=True)
        adf = run_adf(series, regression=adf_regression)
        pval = adf["p-value"]
        stat = adf["ADF Statistic"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_html("ADF Statistic", f"{stat:.4f}", "test statistic",
                                    "green" if stat < adf["Critical Values"]["5%"] else "red"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_html("p-value", f"{pval:.4f}", "AIC lag selection",
                                    "green" if pval < sig_level else "gold"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_html("Lags Used", str(adf["Lags Used"]), "optimal lags"),
                        unsafe_allow_html=True)
        with c4:
            cv5 = adf["Critical Values"]["5%"]
            st.markdown(metric_html("CV (5%)", f"{cv5:.4f}", "critical value"),
                        unsafe_allow_html=True)

        # Interpretation
        if pval < sig_level:
            st.markdown(f"""
            <div class="interpret-stationary">
              ✅ &nbsp;<strong>STATIONARY</strong> — Reject H₀ (unit root) at {int(sig_level*100)}% level
              &nbsp;|&nbsp; p = {pval:.4f} &lt; {sig_level}<br>
              → <em>Series is I(0). Proceed with <strong>OLS</strong> or <strong>VAR</strong>.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="interpret-nonstationary">
              ⚠️ &nbsp;<strong>NON-STATIONARY</strong> — Fail to reject H₀ (unit root) at {int(sig_level*100)}% level
              &nbsp;|&nbsp; p = {pval:.4f} &gt; {sig_level}<br>
              → <em>Consider <strong>first-differencing</strong>, <strong>ARDL bounds test</strong>, or check for cointegration.</em>
            </div>
            """, unsafe_allow_html=True)

        # Critical values table
        with st.expander("📋 Full ADF Critical Values"):
            cv_df = pd.DataFrame(
                {"Level": ["1%", "5%", "10%"],
                 "Critical Value": [adf["Critical Values"]["1%"],
                                    adf["Critical Values"]["5%"],
                                    adf["Critical Values"]["10%"]]})
            st.dataframe(cv_df.style.format({"Critical Value": "{:.4f}"}), use_container_width=True)

        # KPSS
        if run_kpss_flag:
            st.markdown('<p class="section-header">KPSS Test (H₀: Stationary)</p>', unsafe_allow_html=True)
            try:
                kpss_res = run_kpss(series, regression=adf_regression if adf_regression in ["c", "ct"] else "c")
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown(metric_html("KPSS Statistic", f"{kpss_res['KPSS Stat']:.4f}",
                                            "H₀: Stationary",
                                            "green" if kpss_res["p-value"] > sig_level else "red"),
                                unsafe_allow_html=True)
                with kc2:
                    st.markdown(metric_html("p-value", f"{kpss_res['p-value']:.4f}",
                                            "(>0.05 → stationary)",
                                            "green" if kpss_res["p-value"] > sig_level else "red"),
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"KPSS could not run: {e}")

        # ACF/PACF
        st.markdown('<p class="section-header">ACF / PACF Correlograms</p>', unsafe_allow_html=True)
        max_lags = min(24, len(series) // 3)
        st.plotly_chart(acf_pacf_plot(series, nlags=max_lags), use_container_width=True)

        # Download ADF results
        adf_flat = {
            "Variable": label,
            "ADF Statistic": adf["ADF Statistic"],
            "p-value": adf["p-value"],
            "Lags": adf["Lags Used"],
            "CV 1%": adf["Critical Values"]["1%"],
            "CV 5%": adf["Critical Values"]["5%"],
            "CV 10%": adf["Critical Values"]["10%"],
            "Decision": "Stationary" if pval < sig_level else "Non-Stationary",
        }
        st.download_button("⬇ Download ADF Results",
                           results_to_csv(adf_flat),
                           file_name="adf_results.csv",
                           mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Building
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Econometric Model Configuration</p>', unsafe_allow_html=True)

    model_type = st.radio(
        "Select Model",
        ["OLS (Ordinary Least Squares)", "ARDL (Autoregressive Distributed Lag)", "VAR (Vector Autoregression)"],
        horizontal=True,
    )

    # ── OLS ───────────────────────────────────────────────────────────────────
    if model_type.startswith("OLS"):
        st.markdown('<p class="section-header">OLS Configuration</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="latex-box">
        """, unsafe_allow_html=True)
        st.latex(r"y_t = \beta_0 + \sum_{k=1}^{K} \beta_k X_{k,t} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)")
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            dep_var = st.selectbox("Dependent Variable (y)", numeric_cols)
        with c2:
            indep_vars = st.multiselect("Independent Variables (X)", numeric_cols,
                                         default=[c for c in numeric_cols if c != dep_var][:2])

        add_trend = st.checkbox("Include time trend", value=False)
        run_ols = st.button("▶ Estimate OLS")

        if run_ols and dep_var and indep_vars:
            try:
                model_df = df[[dep_var] + indep_vars].dropna()
                y = model_df[dep_var]
                X = model_df[indep_vars].copy()
                if add_trend:
                    X["_trend"] = np.arange(len(X))
                X = add_constant(X)
                model = OLS(y, X).fit()

                st.session_state["ols_model"] = model
                st.session_state["ols_df"] = model_df
                st.session_state["ols_dep"] = dep_var
                st.session_state["ols_indep"] = indep_vars

                # Summary metrics
                st.markdown('<p class="section-header">OLS Estimation Results</p>', unsafe_allow_html=True)
                cm1, cm2, cm3, cm4 = st.columns(4)
                with cm1:
                    st.markdown(metric_html("R²", f"{model.rsquared:.4f}", "coefficient of determination", "blue"), unsafe_allow_html=True)
                with cm2:
                    st.markdown(metric_html("Adj. R²", f"{model.rsquared_adj:.4f}", "penalized for regressors", "gold"), unsafe_allow_html=True)
                with cm3:
                    st.markdown(metric_html("F-statistic", f"{model.fvalue:.3f}", f"p = {model.f_pvalue:.4f}",
                                            "green" if model.f_pvalue < sig_level else "red"), unsafe_allow_html=True)
                with cm4:
                    dw = durbin_watson(model.resid)
                    st.markdown(metric_html("Durbin-Watson", f"{dw:.4f}", "2.0 = no autocorr.",
                                            "green" if 1.5 < dw < 2.5 else "red"), unsafe_allow_html=True)

                # Coefficient table
                coef_df = pd.DataFrame({
                    "Coefficient": model.params,
                    "Std Error": model.bse,
                    "t-stat": model.tvalues,
                    "p-value": model.pvalues,
                    "Signif.": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                                for p in model.pvalues],
                    "[0.025": model.conf_int()[0],
                    "0.975]": model.conf_int()[1],
                })
                st.dataframe(coef_df.style.format({
                    "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
                    "t-stat": "{:.4f}", "p-value": "{:.4f}",
                    "[0.025": "{:.4f}", "0.975]": "{:.4f}",
                }).applymap(lambda v: "color: #69f0ae" if v == "***" else
                             "color: #ffd54f" if v in ["**", "*"] else "", subset=["Signif."]),
                             use_container_width=True)

                # Fitted vs actual
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=model_df.index, y=y.values,
                                              name="Actual", line=dict(color="#4fc3f7", width=2)))
                fig_fit.add_trace(go.Scatter(x=model_df.index, y=model.fittedvalues.values,
                                              name="Fitted", line=dict(color="#ffd54f", width=1.5, dash="dot")))
                fig_fit.update_layout(**PLOTLY_TEMPLATE["layout"],
                                       title=dict(text="Actual vs Fitted", font=dict(size=13, color="#c8d8f0")),
                                       height=320,
                                       margin=dict(l=20, r=20, t=50, b=20),
                                       legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1))
                st.plotly_chart(fig_fit, use_container_width=True)

                # Export
                summary_txt = model.summary().as_text()
                st.download_button("⬇ Download Full OLS Summary",
                                   results_to_txt("OLS", summary_txt),
                                   file_name="ols_results.txt")

            except Exception as e:
                st.error(f"OLS estimation failed: {e}")

    # ── ARDL ──────────────────────────────────────────────────────────────────
    elif model_type.startswith("ARDL"):
        st.markdown('<p class="section-header">ARDL Configuration</p>', unsafe_allow_html=True)
        st.markdown('<div class="latex-box">', unsafe_allow_html=True)
        st.latex(r"\Delta y_t = \alpha_0 + \sum_{i=1}^{p} \phi_i \Delta y_{t-i} + \sum_{j=0}^{q} \beta_j \Delta X_{t-j} + \lambda_1 y_{t-1} + \lambda_2 X_{t-1} + \varepsilon_t")
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            ardl_dep = st.selectbox("Dependent Variable (y)", numeric_cols)
            ardl_lags_y = st.number_input("AR lags (p)", min_value=1, max_value=8, value=1)
        with c2:
            ardl_x = st.multiselect("Exogenous Variables (X)", [c for c in numeric_cols if c != ardl_dep],
                                     default=[c for c in numeric_cols if c != ardl_dep][:1])
            ardl_lags_x = st.number_input("Distributed lags (q)", min_value=0, max_value=8, value=1)

        run_ardl = st.button("▶ Estimate ARDL")

        if run_ardl and ardl_dep and ardl_x:
            try:
                model_df = df[[ardl_dep] + ardl_x].dropna()
                y = model_df[ardl_dep]
                X = model_df[ardl_x]
                order = {col: ardl_lags_x for col in ardl_x}
                ardl_model = ARDL(y, lags=ardl_lags_y, exog=X, order=order).fit()

                st.session_state["ardl_model"] = ardl_model
                st.session_state["ardl_df"] = model_df
                st.session_state["ardl_dep"] = ardl_dep

                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    st.markdown(metric_html("R²", f"{ardl_model.rsquared:.4f}", "ARDL fit"), unsafe_allow_html=True)
                with cm2:
                    st.markdown(metric_html("AIC", f"{ardl_model.aic:.2f}", "information criterion", "gold"), unsafe_allow_html=True)
                with cm3:
                    st.markdown(metric_html("BIC", f"{ardl_model.bic:.2f}", "Bayesian criterion"), unsafe_allow_html=True)

                coef_df = pd.DataFrame({
                    "Coefficient": ardl_model.params,
                    "Std Error": ardl_model.bse,
                    "t-stat": ardl_model.tvalues,
                    "p-value": ardl_model.pvalues,
                })
                st.dataframe(coef_df.style.format("{:.4f}"), use_container_width=True)

                summary_txt = ardl_model.summary().as_text()
                st.download_button("⬇ Download ARDL Summary",
                                   results_to_txt("ARDL", summary_txt),
                                   file_name="ardl_results.txt")

            except Exception as e:
                st.error(f"ARDL estimation failed: {e}")

    # ── VAR ───────────────────────────────────────────────────────────────────
    elif model_type.startswith("VAR"):
        st.markdown('<p class="section-header">VAR Configuration</p>', unsafe_allow_html=True)
        st.markdown('<div class="latex-box">', unsafe_allow_html=True)
        st.latex(r"\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})")
        st.markdown("</div>", unsafe_allow_html=True)

        var_vars = st.multiselect("Select Variables for VAR (min 2)", numeric_cols,
                                   default=numeric_cols[:min(3, len(numeric_cols))])
        var_lags = st.number_input("Max Lags", min_value=1, max_value=12, value=4)
        var_ic = st.selectbox("Lag selection criterion", ["aic", "bic", "hqic", "fpe"])

        run_var = st.button("▶ Estimate VAR")

        if run_var and len(var_vars) >= 2:
            try:
                var_df = df[var_vars].dropna()
                var_model = VAR(var_df)
                results = var_model.select_order(maxlags=var_lags)
                best_lag = getattr(results, var_ic)

                st.info(f"Optimal lag by {var_ic.upper()}: **{best_lag}**")

                fitted = var_model.fit(best_lag)
                st.session_state["var_model"] = fitted
                st.session_state["var_df"] = var_df
                st.session_state["var_vars"] = var_vars

                # Model summary metrics
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    st.markdown(metric_html("Optimal Lags", str(best_lag), f"by {var_ic.upper()}"), unsafe_allow_html=True)
                with cm2:
                    st.markdown(metric_html("AIC", f"{fitted.aic:.2f}", "information criterion", "gold"), unsafe_allow_html=True)
                with cm3:
                    st.markdown(metric_html("Obs.", str(fitted.nobs), "used in estimation"), unsafe_allow_html=True)

                # IRF Plot
                st.markdown('<p class="section-header">Impulse Response Functions (IRF)</p>', unsafe_allow_html=True)
                irf = fitted.irf(10)
                irf_df = pd.DataFrame(
                    {f"{var_vars[i]}→{var_vars[j]}": irf.irfs[:, i, j]
                     for i in range(len(var_vars)) for j in range(len(var_vars))},
                    index=range(11)
                )
                fig_irf = go.Figure()
                for col in irf_df.columns[:6]:
                    fig_irf.add_trace(go.Scatter(x=irf_df.index, y=irf_df[col], name=col, mode="lines+markers",
                                                  marker=dict(size=5)))
                fig_irf.update_layout(**PLOTLY_TEMPLATE["layout"],
                                       title=dict(text="Impulse Response Functions", font=dict(size=13, color="#c8d8f0")),
                                       height=350,
                                       margin=dict(l=20, r=20, t=50, b=20),
                                       legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1))
                st.plotly_chart(fig_irf, use_container_width=True)

                # Forecast
                st.markdown('<p class="section-header">VAR Forecast</p>', unsafe_allow_html=True)
                fc_steps = st.number_input("Forecast periods", min_value=1, max_value=24, value=8)
                fc = fitted.forecast(var_df.values[-best_lag:], steps=fc_steps)
                fc_df = pd.DataFrame(fc, columns=var_vars)

                fig_fc = go.Figure()
                for col in var_vars:
                    fig_fc.add_trace(go.Scatter(y=fc_df[col], name=f"{col} (forecast)",
                                                  line=dict(dash="dot", width=2), mode="lines"))
                fig_fc.update_layout(**PLOTLY_TEMPLATE["layout"],
                                      title=dict(text="VAR Forecast", font=dict(size=13, color="#c8d8f0")),
                                      height=300,
                                      margin=dict(l=20, r=20, t=50, b=20),
                                      legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1))
                st.plotly_chart(fig_fc, use_container_width=True)

                # Export
                st.download_button("⬇ Download VAR Summary",
                                   results_to_txt("VAR", fitted.summary().as_text()),
                                   file_name="var_results.txt")

            except Exception as e:
                st.error(f"VAR estimation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Model Diagnostic Tests</p>', unsafe_allow_html=True)

    # Check if any model has been estimated
    has_ols = "ols_model" in st.session_state
    has_ardl = "ardl_model" in st.session_state
    has_var = "var_model" in st.session_state

    if not (has_ols or has_ardl or has_var):
        st.markdown("""
        <div style='text-align:center; padding:3rem;'>
          <div style='font-size:2.5rem;'>🩺</div>
          <div style='font-family: IBM Plex Mono; color:#4fc3f7; margin-top:0.7rem;'>
            No model estimated yet
          </div>
          <div style='color:#3a5a7a; font-size:0.85rem; margin-top:0.4rem;'>
            Go to Model Building and estimate OLS, ARDL, or VAR first.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        diag_model = st.radio("Select model for diagnostics",
                               (["OLS"] if has_ols else []) +
                               (["ARDL"] if has_ardl else []) +
                               (["VAR"] if has_var else []),
                               horizontal=True)

        if diag_model == "OLS" and has_ols:
            model = st.session_state["ols_model"]
            resid = model.resid.values
            dep = st.session_state["ols_dep"]

            st.markdown(f'<p class="section-header">OLS Residual Diagnostics — {dep}</p>', unsafe_allow_html=True)

            # Tests
            dw = durbin_watson(resid)
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_pval = lb["lb_pvalue"].values[0]
            try:
                bp_stat, bp_pval, _, _ = het_breuschpagan(resid, model.model.exog)
            except Exception:
                bp_stat, bp_pval = np.nan, np.nan

            from scipy import stats as sp_stats
            _, jb_pval = sp_stats.jarque_bera(resid)

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(metric_html("Durbin-Watson", f"{dw:.4f}", "autocorrelation test",
                                        "green" if 1.5 < dw < 2.5 else "red"), unsafe_allow_html=True)
            with d2:
                st.markdown(metric_html("Ljung-Box (p)", f"{lb_pval:.4f}", "serial correlation",
                                        "green" if lb_pval > sig_level else "red"), unsafe_allow_html=True)
            with d3:
                st.markdown(metric_html("Breusch-Pagan (p)", f"{bp_pval:.4f}" if not np.isnan(bp_pval) else "N/A",
                                        "heteroskedasticity",
                                        "green" if (not np.isnan(bp_pval) and bp_pval > sig_level) else "red"),
                            unsafe_allow_html=True)
            with d4:
                st.markdown(metric_html("Jarque-Bera (p)", f"{jb_pval:.4f}", "normality of residuals",
                                        "green" if jb_pval > sig_level else "red"), unsafe_allow_html=True)

            st.plotly_chart(residual_diagnostics_plot(resid), use_container_width=True)

            # Interpretation guide
            st.markdown('<p class="section-header">Diagnostic Interpretation Guide</p>', unsafe_allow_html=True)
            guide_data = {
                "Test": ["Durbin-Watson", "Ljung-Box", "Breusch-Pagan", "Jarque-Bera"],
                "H₀": ["No autocorrelation", "No serial correlation", "Homoskedasticity", "Normality"],
                "Good range": ["1.5 – 2.5", "p > 0.05", "p > 0.05", "p > 0.05"],
                "Your result": [
                    f"{dw:.4f} {'✅' if 1.5<dw<2.5 else '❌'}",
                    f"{lb_pval:.4f} {'✅' if lb_pval>sig_level else '❌'}",
                    f"{bp_pval:.4f} {'✅' if (not np.isnan(bp_pval) and bp_pval>sig_level) else '❌'}",
                    f"{jb_pval:.4f} {'✅' if jb_pval>sig_level else '❌'}",
                ]
            }
            st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

            # Variance Inflation Factor
            st.markdown('<p class="section-header">Multicollinearity — VIF</p>', unsafe_allow_html=True)
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X_cols = model.model.exog_names
            exog = model.model.exog
            try:
                vif_data = pd.DataFrame({
                    "Variable": X_cols,
                    "VIF": [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
                })
                vif_data["Flag"] = vif_data["VIF"].apply(lambda v: "🔴 High" if v > 10 else "🟡 Moderate" if v > 5 else "✅ OK")
                st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"VIF could not be computed: {e}")

            # Download diagnostics
            diag_dict = {
                "Durbin-Watson": dw,
                "LjungBox_p10": lb_pval,
                "BreuschPagan_p": bp_pval,
                "JarqueBera_p": jb_pval,
                "R2": model.rsquared,
                "Adj_R2": model.rsquared_adj,
                "F_stat": model.fvalue,
                "F_pvalue": model.f_pvalue,
            }
            st.download_button("⬇ Download Diagnostics CSV",
                               results_to_csv(diag_dict),
                               file_name="ols_diagnostics.csv",
                               mime="text/csv")

        elif diag_model == "ARDL" and has_ardl:
            model = st.session_state["ardl_model"]
            resid = model.resid.values
            dep = st.session_state["ardl_dep"]

            st.markdown(f'<p class="section-header">ARDL Residual Diagnostics — {dep}</p>', unsafe_allow_html=True)

            dw = durbin_watson(resid)
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_pval = lb["lb_pvalue"].values[0]
            from scipy import stats as sp_stats
            _, jb_pval = sp_stats.jarque_bera(resid)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(metric_html("Durbin-Watson", f"{dw:.4f}", "autocorrelation",
                                        "green" if 1.5<dw<2.5 else "red"), unsafe_allow_html=True)
            with d2:
                st.markdown(metric_html("Ljung-Box (p)", f"{lb_pval:.4f}", "lag 10",
                                        "green" if lb_pval>sig_level else "red"), unsafe_allow_html=True)
            with d3:
                st.markdown(metric_html("Jarque-Bera (p)", f"{jb_pval:.4f}", "normality",
                                        "green" if jb_pval>sig_level else "red"), unsafe_allow_html=True)

            st.plotly_chart(residual_diagnostics_plot(resid), use_container_width=True)

        elif diag_model == "VAR" and has_var:
            fitted = st.session_state["var_model"]
            var_vars = st.session_state["var_vars"]
            st.markdown('<p class="section-header">VAR Stability & Diagnostics</p>', unsafe_allow_html=True)

            # Stability check
            roots = fitted.roots
            is_stable = all(np.abs(r) < 1 for r in roots)
            st.markdown(
                f'<div class="{"interpret-stationary" if is_stable else "interpret-nonstationary"}">'
                f'{"✅ VAR is STABLE — all eigenvalue moduli < 1" if is_stable else "⚠️ VAR may be UNSTABLE — some roots ≥ 1"}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Eigenvalue plot
            fig_roots = go.Figure()
            theta = np.linspace(0, 2*np.pi, 200)
            fig_roots.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta),
                                            mode="lines", line=dict(color="#1e3a5f", width=1),
                                            name="Unit Circle", showlegend=True))
            fig_roots.add_trace(go.Scatter(x=roots.real, y=roots.imag,
                                            mode="markers",
                                            marker=dict(color="#4fc3f7", size=10, symbol="x"),
                                            name="VAR Roots"))
            fig_roots.update_layout(**PLOTLY_TEMPLATE["layout"],
                                     title=dict(text="VAR Stability — Eigenvalue Plot", font=dict(size=13, color="#c8d8f0")),
                                     height=380,
                                     margin=dict(l=20, r=20, t=50, b=20),
                                     xaxis_title="Real", yaxis_title="Imaginary",
                                     yaxis=dict(scaleanchor="x", scaleratio=1,
                                                gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                                     legend=dict(bgcolor="#0d1526", bordercolor="#1e3a5f", borderwidth=1))
            st.plotly_chart(fig_roots, use_container_width=True)

            # Granger causality
            st.markdown('<p class="section-header">Granger Causality Tests</p>', unsafe_allow_html=True)
            var_df = st.session_state["var_df"]
            lag = fitted.k_ar
            gc_rows = []
            for caused in var_vars:
                for causing in var_vars:
                    if caused != causing:
                        try:
                            gc = fitted.test_causality(caused, causing, kind="f")
                            gc_rows.append({
                                "Causing": causing,
                                "Caused": caused,
                                "F-stat": round(gc.test_statistic, 4),
                                "p-value": round(gc.pvalue, 4),
                                "Decision": "Granger-Causes ✅" if gc.pvalue < sig_level else "No causality",
                            })
                        except Exception:
                            pass
            if gc_rows:
                gc_df = pd.DataFrame(gc_rows)
                st.dataframe(gc_df, use_container_width=True, hide_index=True)