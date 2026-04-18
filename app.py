"""
╔══════════════════════════════════════════════════════════════════╗
║      MACRO-FINANCIAL ANALYTICS SUITE  ·  Professional Edition   ║
║      Lead UI/UX + Senior Quant Dev  ·  Single-file app.py       ║
╚══════════════════════════════════════════════════════════════════╝
Dependencies:
    pip install streamlit pandas numpy statsmodels plotly scipy
    pip install openpyxl pmdarima
"""
 
import warnings, io
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
 
from statsmodels.tsa.stattools   import adfuller, kpss, acf, pacf
from statsmodels.tsa.ardl        import ARDL
from statsmodels.tsa.api         import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools     import add_constant
from statsmodels.stats.stattools  import durbin_watson
from statsmodels.stats.diagnostic import (het_arch, acorr_ljungbox,
                                           het_breuschpagan)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
 
# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro-Financial Analytics Suite",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────────────
# THEME DEFINITIONS
# ─────────────────────────────────────────────────────────────────
THEMES = {
    "Executive Dark": {
        "bg":           "#080E1A",
        "bg2":          "#0D1628",
        "bg3":          "#111E35",
        "card":         "#0F1C30",
        "card_border":  "#1B2E4A",
        "accent":       "#00C9A7",
        "accent2":      "#3B9EFF",
        "accent3":      "#FFB800",
        "text":         "#E8F0FE",
        "text2":        "#8BA5C8",
        "text3":        "#4A6585",
        "success":      "#00C9A7",
        "warning":      "#FFB800",
        "danger":       "#FF5370",
        "plotly":       "plotly_dark",
        "plot_bg":      "#0D1628",
        "plot_paper":   "#080E1A",
        "grid":         "#1B2E4A",
        "font_display": "DM Serif Display",
        "font_body":    "DM Sans",
        "font_mono":    "JetBrains Mono",
        "shadow":       "0 4px 24px rgba(0,0,0,0.5), 0 1px 4px rgba(0,201,167,0.08)",
        "glow":         "0 0 20px rgba(0,201,167,0.15)",
    },
    "Research Light": {
        "bg":           "#F4F6FA",
        "bg2":          "#EAEEF5",
        "bg3":          "#DFE5F0",
        "card":         "#FFFFFF",
        "card_border":  "#D0D9EB",
        "accent":       "#1A56DB",
        "accent2":      "#7E3AF2",
        "accent3":      "#E3A008",
        "text":         "#1A2233",
        "text2":        "#445573",
        "text3":        "#8898AA",
        "success":      "#057A55",
        "warning":      "#B45309",
        "danger":       "#C81E1E",
        "plotly":       "plotly_white",
        "plot_bg":      "#FFFFFF",
        "plot_paper":   "#F4F6FA",
        "grid":         "#E5EAF2",
        "font_display": "DM Serif Display",
        "font_body":    "DM Sans",
        "font_mono":    "JetBrains Mono",
        "shadow":       "0 2px 16px rgba(26,86,219,0.08), 0 1px 3px rgba(0,0,0,0.06)",
        "glow":         "0 0 16px rgba(26,86,219,0.10)",
    },
}
 
# ─────────────────────────────────────────────────────────────────
# SIDEBAR — theme selector + upload
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 MF Analytics Suite")
    theme_name = st.selectbox(
        "🎨 Theme",
        list(THEMES.keys()),
        index=0,
        help="Switch between dark executive and light research modes.",
    )
    T = THEMES[theme_name]
 
    st.markdown("---")
    st.markdown("#### 📂 Data Import")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )
 
    st.markdown("---")
    st.markdown("#### ⚙️ Global Settings")
    sig_level = st.select_slider(
        "Significance Level α",
        options=[0.01, 0.05, 0.10],
        value=0.05,
        format_func=lambda x: f"{int(x*100)}%",
    )
    adf_reg = st.selectbox(
        "ADF Regression",
        ["c", "ct", "nc"],
        format_func=lambda x: {"c":"Constant","ct":"Constant+Trend","nc":"None"}[x],
    )
 
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.68rem;color:{T['text3']};font-family:JetBrains Mono,monospace'>"
        "Powered by Streamlit · Statsmodels<br>Plotly · NumPy · SciPy<br><br>"
        "🇪🇬 Cairo · Professional Edition</div>",
        unsafe_allow_html=True,
    )
 
# ─────────────────────────────────────────────────────────────────
# CSS INJECTION  (theme-aware, full design system)
# ─────────────────────────────────────────────────────────────────
def inject_css(T):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
 
    :root {{
        --bg:          {T['bg']};
        --bg2:         {T['bg2']};
        --bg3:         {T['bg3']};
        --card:        {T['card']};
        --card-border: {T['card_border']};
        --accent:      {T['accent']};
        --accent2:     {T['accent2']};
        --accent3:     {T['accent3']};
        --text:        {T['text']};
        --text2:       {T['text2']};
        --text3:       {T['text3']};
        --success:     {T['success']};
        --warning:     {T['warning']};
        --danger:      {T['danger']};
        --shadow:      {T['shadow']};
        --glow:        {T['glow']};
        --ff-display:  '{T['font_display']}', Georgia, serif;
        --ff-body:     '{T['font_body']}', sans-serif;
        --ff-mono:     '{T['font_mono']}', monospace;
    }}
 
    /* ── Reset & Base ── */
    html, body, [class*="css"] {{
        font-family: var(--ff-body);
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }}
    .stApp {{ background-color: var(--bg) !important; }}
    .main .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; }}
 
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: var(--bg2) !important;
        border-right: 1px solid var(--card-border) !important;
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}
    section[data-testid="stSidebar"] h3 {{
        font-family: var(--ff-display);
        font-size: 1.25rem;
        color: var(--accent) !important;
        letter-spacing: 0.02em;
        padding: 0.5rem 0;
    }}
    section[data-testid="stSidebar"] h4 {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text3) !important;
        margin-top: 0.5rem;
    }}
 
    /* ── Cards ── */
    .mf-card {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.2s;
    }}
    .mf-card:hover {{ box-shadow: var(--shadow), var(--glow); }}
    .mf-card.accent-left::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent), var(--accent2));
        border-radius: 12px 0 0 12px;
    }}
    .mf-card.accent-gold::before {{
        background: linear-gradient(180deg, var(--accent3), var(--warning));
    }}
    .mf-card.accent-danger::before {{
        background: linear-gradient(180deg, var(--danger), #FF8A65);
    }}
 
    /* KPI card */
    .kpi-label {{
        font-family: var(--ff-mono);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--text3);
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-family: var(--ff-mono);
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.72rem;
        color: var(--text3);
        margin-top: 0.3rem;
    }}
    .kpi-delta-pos {{ color: var(--success); font-size: 0.78rem; font-weight: 600; }}
    .kpi-delta-neg {{ color: var(--danger);  font-size: 0.78rem; font-weight: 600; }}
 
    /* Verdict boxes */
    .verdict-stationary {{
        background: color-mix(in srgb, var(--success) 10%, transparent);
        border: 1px solid var(--success);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: var(--success);
        font-family: var(--ff-mono);
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }}
    .verdict-nonstationary {{
        background: color-mix(in srgb, var(--warning) 10%, transparent);
        border: 1px solid var(--warning);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: var(--warning);
        font-family: var(--ff-mono);
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }}
    .verdict-danger {{
        background: color-mix(in srgb, var(--danger) 10%, transparent);
        border: 1px solid var(--danger);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: var(--danger);
        font-family: var(--ff-mono);
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }}
 
    /* Section headers */
    .sec-header {{
        font-family: var(--ff-mono);
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: var(--accent);
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--card-border);
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }}
 
    /* App banner */
    .app-banner {{
        background: linear-gradient(135deg, var(--bg2) 0%, var(--bg3) 100%);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 1.6rem 2.2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .banner-title {{
        font-family: var(--ff-display);
        font-size: 2rem;
        color: var(--text);
        letter-spacing: 0.01em;
        line-height: 1.1;
    }}
    .banner-title span {{ color: var(--accent); }}
    .banner-sub {{
        font-size: 0.83rem;
        color: var(--text2);
        margin-top: 0.4rem;
        font-family: var(--ff-body);
    }}
    .banner-pill {{
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #fff;
        font-family: var(--ff-mono);
        font-size: 0.68rem;
        letter-spacing: 0.07em;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        text-transform: uppercase;
    }}
 
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--bg2);
        border-bottom: 1px solid var(--card-border);
        gap: 0;
        padding: 0 0.5rem;
        border-radius: 10px 10px 0 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: var(--ff-mono);
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        color: var(--text3) !important;
        padding: 0.75rem 1.2rem;
        border-bottom: 2px solid transparent;
        background: transparent;
        transition: color 0.2s;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent;
        padding-top: 1.2rem;
    }}
 
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #fff !important;
        border: none;
        border-radius: 7px;
        font-family: var(--ff-mono);
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s;
        width: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25), var(--glow);
    }}
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #065F46, #047857) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 7px !important;
        font-family: var(--ff-mono) !important;
        font-size: 0.78rem !important;
        width: 100%;
    }}
 
    /* Selects / Inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput input,
    .stTextInput input {{
        background: var(--bg2) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--text) !important;
        border-radius: 7px !important;
    }}
 
    /* Dataframes */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    iframe[title="st.iframe"] {{ border: none; }}
 
    /* LaTeX container */
    .latex-block {{
        background: var(--bg2);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1.2rem 2rem;
        text-align: center;
        margin: 0.8rem 0 1.2rem;
    }}
 
    /* Empty state */
    .empty-state {{
        text-align: center;
        padding: 5rem 2rem;
    }}
    .empty-icon {{ font-size: 3.5rem; margin-bottom: 1rem; }}
    .empty-title {{
        font-family: var(--ff-display);
        font-size: 1.4rem;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }}
    .empty-sub {{ font-size: 0.85rem; color: var(--text3); max-width: 420px; margin: 0 auto; }}
 
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--card-border); border-radius: 4px; }}
 
    /* Expander */
    .streamlit-expanderHeader {{
        background: var(--bg2) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        font-family: var(--ff-mono) !important;
        font-size: 0.78rem !important;
        color: var(--accent) !important;
    }}
    .streamlit-expanderContent {{
        background: var(--bg2) !important;
        border: 1px solid var(--card-border) !important;
        border-top: none !important;
    }}
 
    /* Radio */
    .stRadio label {{ color: var(--text) !important; }}
    .stCheckbox label {{ color: var(--text) !important; }}
 
    /* Metric override */
    [data-testid="metric-container"] {{
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: var(--shadow);
    }}
    </style>
    """, unsafe_allow_html=True)
 
inject_css(T)
 
# ─────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT FACTORY
# ─────────────────────────────────────────────────────────────────
def plo(T, height=360, title="", margin=None):
    m = margin or dict(l=20, r=20, t=50 if title else 30, b=20)
    return dict(
        template=T["plotly"],
        paper_bgcolor=T["plot_paper"],
        plot_bgcolor=T["plot_bg"],
        font=dict(family=T["font_mono"], color=T["text2"], size=11),
        title=dict(text=title, font=dict(size=13, color=T["text"], family=T["font_body"]), x=0.01) if title else {},
        xaxis=dict(gridcolor=T["grid"], linecolor=T["card_border"], zerolinecolor=T["grid"]),
        yaxis=dict(gridcolor=T["grid"], linecolor=T["card_border"], zerolinecolor=T["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=T["card_border"],
                    borderwidth=1, font=dict(size=10)),
        height=height,
        margin=m,
        hovermode="x unified",
    )
 
COLORS = ["#00C9A7","#3B9EFF","#FFB800","#FF5370","#A78BFA","#34D399","#F97316","#EC4899"]
 
# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub=None, delta=None, accent="accent-left"):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.3f}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="mf-card {accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {sub_html}{delta_html}
    </div>"""
 
def sec(label):
    return f'<p class="sec-header">{label}</p>'
 
def verdict_html(stationary, pval, sig):
    if stationary:
        return f"""<div class="verdict-stationary">
        ✅ &nbsp;<strong>STATIONARY</strong> — Reject H₀ at {int(sig*100)}% level &nbsp;|&nbsp; p = {pval:.4f}<br>
        → <em>Proceed with <strong>OLS</strong> or <strong>VAR</strong></em>
        </div>"""
    else:
        return f"""<div class="verdict-nonstationary">
        ⚠️ &nbsp;<strong>NON-STATIONARY</strong> — Fail to reject H₀ &nbsp;|&nbsp; p = {pval:.4f}<br>
        → <em>Consider <strong>ARDL Bounds</strong>, first-differencing, or cointegration</em>
        </div>"""
 
def load_data(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    # Ghost-row fix
    df = df.dropna(how="all").reset_index(drop=True)
    return df
 
def detect_date(df):
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["year","date","time","period","yr","quarter"]):
            try:
                parsed = pd.to_datetime(df[col], format="%Y", errors="coerce")
                if parsed.notna().sum() > len(df)*0.5:
                    return col, parsed
            except:
                pass
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df)*0.5:
                    return col, parsed
            except:
                pass
    return None, None
 
def to_float(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols)
 
def run_adf(series, reg="c"):
    res = adfuller(series.dropna(), regression=reg, autolag="AIC")
    return {"stat": res[0], "pval": res[1], "lags": res[2], "cv": res[4]}
 
def run_kpss(series, reg="c"):
    try:
        reg2 = reg if reg in ["c","ct"] else "c"
        stat, pval, lags, cv = kpss(series.dropna(), regression=reg2, nlags="auto")
        return {"stat": stat, "pval": pval, "lags": lags, "cv": cv}
    except:
        return None
 
def arch_lm(resid, lags=5):
    try:
        lm, lm_p, fstat, f_p = het_arch(resid, nlags=lags)
        return lm, lm_p
    except:
        return np.nan, np.nan
 
def cusum_series(resid):
    n = len(resid)
    s = np.std(resid, ddof=1)
    cs = np.cumsum(resid) / (s * np.sqrt(n)) if s > 0 else np.cumsum(resid)
    bounds = [0.948 * np.sqrt(t/n + 2*t/n) for t in range(1, n+1)]
    return cs, np.array(bounds)
 
def cusumsq_series(resid):
    sq = resid**2
    cs = np.cumsum(sq) / (np.sum(sq) + 1e-12)
    n = len(resid)
    bounds = np.sqrt(np.linspace(0.04/n, 1-0.04/n, n) * 0.04 + np.linspace(0,1,n))
    return cs, bounds
 
def auto_arima_aic(series, max_p=4, max_q=4, max_d=2):
    """Simple grid-search AIC-based ARIMA selection."""
    best_aic, best_order = np.inf, (1,1,0)
    for d in range(max_d+1):
        for p in range(max_p+1):
            for q in range(max_q+1):
                try:
                    m = ARIMA(series, order=(p,d,q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order = m.aic, (p,d,q)
                except:
                    pass
    return best_order, best_aic
 
def results_csv(d):
    return pd.DataFrame([{"Metric":k,"Value":v} for k,v in d.items()]).to_csv(index=False).encode()
 
def results_txt(header, body):
    return (f"{'='*70}\n  {header}\n{'='*70}\n\n{body}\n").encode()
 
# ─────────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-banner">
  <div>
    <div class="banner-title">Macro<span>-Financial</span> Analytics Suite</div>
    <div class="banner-sub">
      Stationarity · Regression · ARDL · VAR · ARIMA Forecasting · Diagnostics
    </div>
  </div>
  <div>
    <span class="banner-pill">{theme_name}</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────
# NO DATA STATE
# ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-icon">📊</div>
      <div class="empty-title">Upload a Dataset to Begin</div>
      <div class="empty-sub">
        Supports CSV and Excel. The suite auto-detects date columns, 
        cleans ghost rows, and guides you through the full econometric workflow.
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Feature showcase
    c1,c2,c3,c4,c5 = st.columns(5)
    feats = [
        ("🔬","Unit Root Tests","ADF + KPSS with auto-verdict"),
        ("📐","3 Model Classes","OLS · ARDL · VAR with LaTeX"),
        ("🩺","Full Diagnostics","ARCH-LM · CUSUM · VIF"),
        ("🔮","ARIMA Forecast","Auto-selection + CI shading"),
        ("📤","Export Ready","CSV + TXT summary reports"),
    ]
    for col,(icon,title,sub) in zip([c1,c2,c3,c4,c5],feats):
        with col:
            st.markdown(f"""
            <div class="mf-card accent-left" style="text-align:center">
              <div style="font-size:1.8rem">{icon}</div>
              <div style="font-family:var(--ff-body);font-weight:600;color:var(--text);
                          margin:0.5rem 0 0.2rem;font-size:0.9rem">{title}</div>
              <div style="font-size:0.75rem;color:var(--text3)">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()
 
# ─────────────────────────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
try:
    df_raw = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}"); st.stop()
 
date_col, date_series = detect_date(df_raw)
if date_col:
    df_raw.index = date_series
    df_raw = df_raw.drop(columns=[date_col])
 
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found. Check your file."); st.stop()
 
# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Workspace",
    "🔬  Unit Root Tests",
    "📐  Econometric Models",
    "🩺  Diagnostics",
    "🔮  ARIMA Forecast",
])
 
# ══════════════════════════════════════════════════════════════════
# TAB 1 — WORKSPACE
# ══════════════════════════════════════════════════════════════════
with tab1:
    # Dataset KPIs
    st.markdown(sec("Dataset Overview"), unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi_card("Observations",f"{len(df_raw):,}","total rows"), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card("Variables",str(len(numeric_cols)),"numeric columns","accent-gold"), unsafe_allow_html=True)
    with k3:
        miss = df_raw[numeric_cols].isna().mean().mean()*100
        acc = "accent-danger" if miss>5 else "accent-left"
        st.markdown(kpi_card("Missing Data",f"{miss:.1f}%","avg across cols",accent=acc), unsafe_allow_html=True)
    with k4:
        if isinstance(df_raw.index, pd.DatetimeIndex):
            rng = f"{df_raw.index.min().year}–{df_raw.index.max().year}"
        else:
            rng = f"0–{len(df_raw)-1}"
        st.markdown(kpi_card("Index Range",rng,"time coverage"), unsafe_allow_html=True)
 
    st.markdown("---")
 
    # Time-series chart
    left, right = st.columns([3,1])
    with right:
        sel_ts = st.multiselect("Plot variables",numeric_cols,
                                default=numeric_cols[:min(3,len(numeric_cols))],
                                key="ts_sel")
        normalize = st.checkbox("Normalize (z-score)", value=False)
    with left:
        if sel_ts:
            plot_df = df_raw[sel_ts].copy()
            if normalize:
                plot_df = (plot_df - plot_df.mean()) / plot_df.std()
            fig = go.Figure()
            for i,c in enumerate(sel_ts):
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[c], name=c,
                                          line=dict(color=COLORS[i%len(COLORS)],width=2),
                                          mode="lines"))
            fig.update_layout(**plo(T, height=340,
                                    title="Time Series — " + (", ".join(sel_ts[:3])+("…" if len(sel_ts)>3 else ""))))
            st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("---")
 
    # Correlation heatmap + distribution
    col_h, col_d = st.columns([1,1])
    with col_h:
        st.markdown(sec("Correlation Heatmap"), unsafe_allow_html=True)
        hm_cols = st.multiselect("Variables",numeric_cols,
                                  default=numeric_cols[:min(6,len(numeric_cols))], key="hm_sel")
        if len(hm_cols)>=2:
            corr = df_raw[hm_cols].corr()
            fig_hm = px.imshow(corr, text_auto=".2f",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                aspect="auto")
            fig_hm.update_layout(**plo(T, height=340))
            fig_hm.update_coloraxes(colorbar_tickfont_color=T["text2"])
            st.plotly_chart(fig_hm, use_container_width=True)
 
    with col_d:
        st.markdown(sec("Distribution"), unsafe_allow_html=True)
        dist_var = st.selectbox("Variable",numeric_cols, key="dist_var")
        series_d = df_raw[dist_var].dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=series_d, nbinsx=25,
                                         marker_color=COLORS[0], opacity=0.75, name="Freq"))
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series_d)
        xs = np.linspace(series_d.min(), series_d.max(), 200)
        scale = len(series_d)*(series_d.max()-series_d.min())/25
        fig_dist.add_trace(go.Scatter(x=xs, y=kde(xs)*scale, mode="lines",
                                       line=dict(color=COLORS[2],width=2.5), name="KDE"))
        fig_dist.update_layout(**plo(T, height=340, title=f"Distribution: {dist_var}"))
        st.plotly_chart(fig_dist, use_container_width=True)
 
    # Summary stats
    with st.expander("📊 Descriptive Statistics"):
        st.dataframe(df_raw[numeric_cols].describe().T.style.format("{:.4f}"),
                     use_container_width=True)
    with st.expander("📄 Raw Data"):
        st.dataframe(df_raw[numeric_cols].style.format("{:.4f}"), use_container_width=True)
        st.download_button("⬇ Download CSV", df_raw.to_csv().encode(),
                           "dataset.csv","text/csv")
 
 
# ══════════════════════════════════════════════════════════════════
# TAB 2 — UNIT ROOT TESTS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(sec("Augmented Dickey-Fuller & KPSS Tests"), unsafe_allow_html=True)
 
    left, right = st.columns([1,2])
    with left:
        st.markdown(f'<div class="mf-card accent-left">', unsafe_allow_html=True)
        ur_var = st.selectbox("Variable", numeric_cols, key="ur_var")
        diff_order = st.radio("Differencing", [0,1,2],
                               format_func=lambda x: ["I(0) — Level","I(1) — First Diff","I(2) — Second Diff"][x])
        run_ur = st.button("▶ Run Unit Root Tests", key="run_ur")
        st.markdown("</div>", unsafe_allow_html=True)
 
    if run_ur:
        series = df_raw[ur_var].copy()
        series = pd.to_numeric(series, errors="coerce").dropna()
        if diff_order==1: series = series.diff().dropna()
        elif diff_order==2: series = series.diff().diff().dropna()
        label = ur_var + ["","  Δ","  Δ²"][diff_order]
 
        # Plot
        with right:
            fig_ur = go.Figure()
            fig_ur.add_trace(go.Scatter(x=list(range(len(series))), y=series.values,
                                         line=dict(color=COLORS[0],width=2), name=label))
            fig_ur.update_layout(**plo(T, height=220, title=f"Series: {label}"))
            st.plotly_chart(fig_ur, use_container_width=True)
 
        # ADF
        adf = run_adf(series, adf_reg)
        kps = run_kpss(series, adf_reg)
 
        st.markdown(sec("ADF Test — H₀: Unit Root (non-stationary)"), unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        adf_stationary = adf["pval"] < sig_level
        with a1: st.markdown(kpi_card("ADF Statistic",f"{adf['stat']:.4f}","test statistic",
                                       accent="accent-left" if adf_stationary else "accent-danger"), unsafe_allow_html=True)
        with a2: st.markdown(kpi_card("p-value",f"{adf['pval']:.4f}",f"α = {sig_level}",
                                       accent="accent-left" if adf_stationary else "accent-gold"), unsafe_allow_html=True)
        with a3: st.markdown(kpi_card("Lags (AIC)",str(adf["lags"]),"auto-selected"), unsafe_allow_html=True)
        with a4: st.markdown(kpi_card("CV 5%",f"{adf['cv']['5%']:.4f}","critical value"), unsafe_allow_html=True)
 
        st.markdown(verdict_html(adf_stationary, adf["pval"], sig_level), unsafe_allow_html=True)
 
        # KPSS
        if kps:
            st.markdown(sec("KPSS Test — H₀: Stationary"), unsafe_allow_html=True)
            kpss_stationary = kps["pval"] > sig_level
            k1,k2,k3 = st.columns(3)
            with k1: st.markdown(kpi_card("KPSS Statistic",f"{kps['stat']:.4f}","",
                                           accent="accent-left" if kpss_stationary else "accent-danger"), unsafe_allow_html=True)
            with k2: st.markdown(kpi_card("p-value",f"{kps['pval']:.4f}","(>α → stationary)",
                                           accent="accent-left" if kpss_stationary else "accent-gold"), unsafe_allow_html=True)
            with k3: st.markdown(kpi_card("Lags",str(kps["lags"]),"bandwidth"), unsafe_allow_html=True)
 
            kpss_txt = "✅ KPSS confirms: Stationary" if kpss_stationary else "⚠️ KPSS also detects: Non-Stationary"
            cls = "verdict-stationary" if kpss_stationary else "verdict-nonstationary"
            st.markdown(f'<div class="{cls}">{kpss_txt}</div>', unsafe_allow_html=True)
 
        # ACF/PACF
        st.markdown(sec("ACF / PACF Correlograms"), unsafe_allow_html=True)
        nlags = min(24, len(series)//4)
        acf_v  = acf(series, nlags=nlags, fft=True)
        pacf_v = pacf(series, nlags=nlags)
        ci = 1.96/np.sqrt(len(series))
        lags_range = list(range(len(acf_v)))
 
        fig_ap = make_subplots(1,2, subplot_titles=["ACF","PACF"])
        for vals, col_idx in [(acf_v,1),(pacf_v,2)]:
            for i,v in enumerate(vals):
                clr = COLORS[0] if abs(v)>ci else T["text3"]
                fig_ap.add_trace(go.Bar(x=[i],y=[v],marker_color=clr,showlegend=False),row=1,col=col_idx)
            fig_ap.add_trace(go.Scatter(x=lags_range,y=[ci]*len(lags_range),mode="lines",
                                         line=dict(dash="dash",color=COLORS[2],width=1),showlegend=False),row=1,col=col_idx)
            fig_ap.add_trace(go.Scatter(x=lags_range,y=[-ci]*len(lags_range),mode="lines",
                                         line=dict(dash="dash",color=COLORS[2],width=1),showlegend=False),row=1,col=col_idx)
        fig_ap.update_layout(**plo(T,height=280))
        for ann in fig_ap.layout.annotations: ann.font.color=T["text"]
        st.plotly_chart(fig_ap, use_container_width=True)
 
        # Export
        export_d = {"Variable":label,"ADF_stat":adf["stat"],"ADF_pval":adf["pval"],
                    "ADF_lags":adf["lags"],"ADF_cv5":adf["cv"]["5%"],
                    "Decision": "Stationary" if adf_stationary else "Non-Stationary"}
        st.download_button("⬇ Export ADF Results", results_csv(export_d),
                           "adf_results.csv","text/csv")
 
 
# ══════════════════════════════════════════════════════════════════
# TAB 3 — ECONOMETRIC MODELS
# ══════════════════════════════════════════════════════════════════
with tab3:
    model_sel = st.radio("Model",
                          ["OLS — Ordinary Least Squares",
                           "ARDL — Autoregressive Distributed Lag",
                           "VAR — Vector Autoregression"],
                          horizontal=True)
 
    # ── OLS ──────────────────────────────────────────────────────
    if model_sel.startswith("OLS"):
        st.markdown(f'<div class="latex-block">', unsafe_allow_html=True)
        st.latex(r"y_t = \beta_0 + \sum_{k=1}^{K}\beta_k X_{k,t} + \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma^2)")
        st.markdown("</div>", unsafe_allow_html=True)
 
        c1,c2 = st.columns(2)
        with c1: dep = st.selectbox("Dependent (y)", numeric_cols)
        with c2: indep = st.multiselect("Regressors (X)", numeric_cols,
                                         default=[c for c in numeric_cols if c!=dep][:2])
        trend = st.checkbox("Include linear time trend")
        run_ols = st.button("▶ Estimate OLS")
 
        if run_ols and dep and indep:
            mdf = to_float(df_raw[[dep]+indep].copy(), [dep]+indep)
            y = mdf[dep]; X = mdf[indep].copy()
            if trend: X["__trend"] = np.arange(len(X))
            X = add_constant(X)
            try:
                res = OLS(y,X).fit()
                st.session_state["ols_res"] = res
                st.session_state["ols_dep"] = dep
 
                st.markdown(sec("Estimation Results"), unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                dw = durbin_watson(res.resid)
                with m1: st.markdown(kpi_card("R²",f"{res.rsquared:.4f}","coefficient of det."), unsafe_allow_html=True)
                with m2: st.markdown(kpi_card("Adj. R²",f"{res.rsquared_adj:.4f}","penalized fit","accent-gold"), unsafe_allow_html=True)
                with m3: st.markdown(kpi_card("F-stat",f"{res.fvalue:.3f}",f"p={res.f_pvalue:.4f}",
                                               accent="accent-left" if res.f_pvalue<sig_level else "accent-danger"), unsafe_allow_html=True)
                with m4: st.markdown(kpi_card("Durbin-Watson",f"{dw:.4f}","2.0=no autocorr.",
                                               accent="accent-left" if 1.5<dw<2.5 else "accent-gold"), unsafe_allow_html=True)
 
                coef = pd.DataFrame({
                    "Coeff":res.params,"Std Err":res.bse,
                    "t-stat":res.tvalues,"p-value":res.pvalues,
                    "Sig":[("***" if p<.01 else "**" if p<.05 else "*" if p<.1 else "")
                           for p in res.pvalues],
                    "CI 2.5%":res.conf_int()[0],"CI 97.5%":res.conf_int()[1]
                })
                st.dataframe(coef.style.format({c:"{:.4f}" for c in coef.columns if c!="Sig"}),
                             use_container_width=True)
 
                # Actual vs Fitted
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(y=y.values,name="Actual",
                                              line=dict(color=COLORS[0],width=2)))
                fig_fit.add_trace(go.Scatter(y=res.fittedvalues.values,name="Fitted",
                                              line=dict(color=COLORS[2],width=1.8,dash="dot")))
                fig_fit.update_layout(**plo(T,height=300,title="Actual vs Fitted"))
                st.plotly_chart(fig_fit, use_container_width=True)
 
                st.download_button("⬇ Export OLS Summary",
                                   results_txt("OLS Results",res.summary().as_text()),
                                   "ols_summary.txt")
            except Exception as e:
                st.error(f"OLS failed: {e}")
 
    # ── ARDL ─────────────────────────────────────────────────────
    elif model_sel.startswith("ARDL"):
        st.markdown('<div class="latex-block">', unsafe_allow_html=True)
        st.latex(r"\Delta y_t=\alpha_0+\sum_{i=1}^{p}\phi_i\Delta y_{t-i}+\sum_{j=0}^{q}\beta_j\Delta X_{t-j}+\lambda_1 y_{t-1}+\lambda_2 X_{t-1}+\varepsilon_t")
        st.markdown("</div>", unsafe_allow_html=True)
 
        c1,c2,c3,c4 = st.columns(4)
        with c1: ardl_dep = st.selectbox("Dependent (y)", numeric_cols, key="ardl_dep")
        with c2: ardl_x = st.multiselect("Exogenous (X)", [c for c in numeric_cols if c!=ardl_dep],
                                          default=[c for c in numeric_cols if c!=ardl_dep][:1])
        with c3: ardl_p = st.number_input("AR lags p", 1, 8, 1)
        with c4: ardl_q = st.number_input("DL lags q", 0, 8, 1)
        run_ardl = st.button("▶ Estimate ARDL")
 
        if run_ardl and ardl_dep and ardl_x:
            mdf = to_float(df_raw[[ardl_dep]+ardl_x].copy(), [ardl_dep]+ardl_x)
            try:
                order = {c: ardl_q for c in ardl_x}
                res = ARDL(mdf[ardl_dep], lags=ardl_p, exog=mdf[ardl_x], order=order).fit()
                st.session_state["ardl_res"] = res
                st.session_state["ardl_dep"] = ardl_dep
 
                a1,a2,a3 = st.columns(3)
                with a1: st.markdown(kpi_card("R²",f"{res.rsquared:.4f}"), unsafe_allow_html=True)
                with a2: st.markdown(kpi_card("AIC",f"{res.aic:.2f}","information criterion","accent-gold"), unsafe_allow_html=True)
                with a3: st.markdown(kpi_card("BIC",f"{res.bic:.2f}"), unsafe_allow_html=True)
 
                st.dataframe(pd.DataFrame({
                    "Coeff":res.params,"Std Err":res.bse,"t":res.tvalues,"p":res.pvalues
                }).style.format("{:.4f}"), use_container_width=True)
 
                st.download_button("⬇ Export ARDL Summary",
                                   results_txt("ARDL Results",res.summary().as_text()),
                                   "ardl_summary.txt")
            except Exception as e:
                st.error(f"ARDL failed: {e}")
 
    # ── VAR ──────────────────────────────────────────────────────
    else:
        st.markdown('<div class="latex-block">', unsafe_allow_html=True)
        st.latex(r"\mathbf{y}_t=\mathbf{c}+\sum_{i=1}^{p}\mathbf{A}_i\mathbf{y}_{t-i}+\boldsymbol{\varepsilon}_t,\quad\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma})")
        st.markdown("</div>", unsafe_allow_html=True)
 
        c1,c2,c3 = st.columns(3)
        with c1: var_vars = st.multiselect("VAR variables (≥2)", numeric_cols,
                                            default=numeric_cols[:min(3,len(numeric_cols))])
        with c2: var_max = st.number_input("Max lags", 1, 12, 4)
        with c3: var_ic  = st.selectbox("IC criterion", ["aic","bic","hqic","fpe"])
        run_var = st.button("▶ Estimate VAR")
 
        if run_var and len(var_vars)>=2:
            mdf = to_float(df_raw[var_vars].copy(), var_vars)
            try:
                vm = VAR(mdf)
                sel = vm.select_order(maxlags=var_max)
                best_lag = max(1, getattr(sel, var_ic))
                res = vm.fit(best_lag)
                st.session_state["var_res"] = res
                st.session_state["var_vars"] = var_vars
                st.session_state["var_df"] = mdf
 
                v1,v2,v3 = st.columns(3)
                with v1: st.markdown(kpi_card("Optimal Lags",str(best_lag),f"by {var_ic.upper()}"), unsafe_allow_html=True)
                with v2: st.markdown(kpi_card("AIC",f"{res.aic:.2f}","","accent-gold"), unsafe_allow_html=True)
                with v3: st.markdown(kpi_card("Obs.",str(res.nobs)), unsafe_allow_html=True)
 
                # IRF
                irf = res.irf(10)
                fig_irf = go.Figure()
                for i in range(len(var_vars)):
                    for j in range(len(var_vars)):
                        fig_irf.add_trace(go.Scatter(
                            y=irf.irfs[:,i,j],
                            name=f"{var_vars[i]}→{var_vars[j]}",
                            line=dict(width=1.8,color=COLORS[(i*len(var_vars)+j)%len(COLORS)])))
                fig_irf.update_layout(**plo(T,height=320,title="Impulse Response Functions"))
                st.plotly_chart(fig_irf, use_container_width=True)
 
                # Granger
                st.markdown(sec("Granger Causality"), unsafe_allow_html=True)
                gc_rows=[]
                for caused in var_vars:
                    for causing in var_vars:
                        if caused!=causing:
                            try:
                                gc=res.test_causality(caused,causing,kind="f")
                                gc_rows.append({"Causing":causing,"Caused":caused,
                                                "F":round(gc.test_statistic,4),
                                                "p":round(gc.pvalue,4),
                                                "Decision":"✅ Causes" if gc.pvalue<sig_level else "✗ No causality"})
                            except: pass
                if gc_rows: st.dataframe(pd.DataFrame(gc_rows),use_container_width=True,hide_index=True)
 
                st.download_button("⬇ Export VAR Summary",
                                   results_txt("VAR Results",res.summary().as_text()),
                                   "var_summary.txt")
            except Exception as e:
                st.error(f"VAR failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════
# TAB 4 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(sec("Post-Estimation Diagnostic Tests"), unsafe_allow_html=True)
 
    has_ols  = "ols_res"  in st.session_state
    has_ardl = "ardl_res" in st.session_state
    has_var  = "var_res"  in st.session_state
 
    if not (has_ols or has_ardl or has_var):
        st.markdown(f"""<div class="empty-state">
          <div class="empty-icon">🩺</div>
          <div class="empty-title">Estimate a Model First</div>
          <div class="empty-sub">Head to the Econometric Models tab, run OLS, ARDL, or VAR, then return here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        options = (["OLS"] if has_ols else [])+(["ARDL"] if has_ardl else [])+(["VAR"] if has_var else [])
        diag_model = st.radio("Diagnostics for:", options, horizontal=True)
 
        if diag_model in ["OLS","ARDL"]:
            res = st.session_state["ols_res"] if diag_model=="OLS" else st.session_state["ardl_res"]
            resid = res.resid.values
 
            # Core stats
            dw   = durbin_watson(resid)
            lb   = acorr_ljungbox(resid,lags=[10],return_df=True)
            lb_p = lb["lb_pvalue"].values[0]
            arch_stat, arch_p = arch_lm(resid)
            _, jb_p = sp_stats.jarque_bera(resid)
            try:
                _, bp_p, _, _ = het_breuschpagan(resid, res.model.exog)
            except: bp_p = np.nan
 
            st.markdown(sec("Core Diagnostic Statistics"), unsafe_allow_html=True)
            d1,d2,d3,d4,d5 = st.columns(5)
            tests = [
                ("Durbin-Watson",f"{dw:.4f}","No autocorr. ≈ 2", 1.5<dw<2.5),
                ("Ljung-Box p","—" if np.isnan(lb_p) else f"{lb_p:.4f}","Lag 10", lb_p>sig_level),
                ("Breusch-Pagan p","—" if np.isnan(bp_p) else f"{bp_p:.4f}","Homoskedasticity", (not np.isnan(bp_p)) and bp_p>sig_level),
                ("ARCH-LM p","—" if np.isnan(arch_p) else f"{arch_p:.4f}","No volatility cluster.", np.isnan(arch_p) or arch_p>sig_level),
                ("Jarque-Bera p",f"{jb_p:.4f}","Normality", jb_p>sig_level),
            ]
            for col, (lbl,val,sub,ok) in zip([d1,d2,d3,d4,d5], tests):
                with col: st.markdown(kpi_card(lbl,val,sub,
                                                accent="accent-left" if ok else "accent-danger"), unsafe_allow_html=True)
 
            # 4-panel residual plot
            st.markdown(sec("Residual Diagnostics"), unsafe_allow_html=True)
            fig_r = make_subplots(2,2,
                                   subplot_titles=["Residuals Over Time","Histogram + KDE",
                                                   "ACF of Residuals","Q-Q Plot"])
            t = list(range(len(resid)))
            ci_r = 1.96/np.sqrt(len(resid))
            acf_r = acf(resid, nlags=20, fft=True)
 
            fig_r.add_trace(go.Scatter(x=t,y=resid,mode="lines",
                                        line=dict(color=COLORS[0],width=1.5),showlegend=False),1,1)
            fig_r.add_trace(go.Scatter(x=t,y=[0]*len(t),mode="lines",
                                        line=dict(color=COLORS[2],dash="dash",width=1),showlegend=False),1,1)
 
            from scipy.stats import gaussian_kde as gkde
            kde2 = gkde(resid)
            xs2 = np.linspace(resid.min(),resid.max(),200)
            fig_r.add_trace(go.Histogram(x=resid,nbinsx=25,marker_color=COLORS[0],
                                          opacity=0.6,showlegend=False),1,2)
            scale2 = len(resid)*(resid.max()-resid.min())/25
            fig_r.add_trace(go.Scatter(x=xs2,y=kde2(xs2)*scale2,mode="lines",
                                        line=dict(color=COLORS[2],width=2),showlegend=False),1,2)
 
            for i,v in enumerate(acf_r):
                fig_r.add_trace(go.Bar(x=[i],y=[v],
                                        marker_color=COLORS[0] if abs(v)>ci_r else T["text3"],
                                        showlegend=False),2,1)
            fig_r.add_trace(go.Scatter(x=list(range(21)),y=[ci_r]*21,mode="lines",
                                        line=dict(dash="dash",color=COLORS[2],width=1),showlegend=False),2,1)
            fig_r.add_trace(go.Scatter(x=list(range(21)),y=[-ci_r]*21,mode="lines",
                                        line=dict(dash="dash",color=COLORS[2],width=1),showlegend=False),2,1)
 
            qq = sp_stats.probplot(resid)
            fig_r.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1],mode="markers",
                                        marker=dict(color=COLORS[0],size=5),showlegend=False),2,2)
            fig_r.add_trace(go.Scatter(x=qq[0][0],
                                        y=qq[1][1]+qq[1][0]*np.array(qq[0][0]),mode="lines",
                                        line=dict(color=COLORS[2],width=1.5),showlegend=False),2,2)
 
            fig_r.update_layout(**plo(T,height=480))
            for ann in fig_r.layout.annotations: ann.font.color=T["text"]
            st.plotly_chart(fig_r, use_container_width=True)
 
            # CUSUM
            st.markdown(sec("CUSUM Structural Stability"), unsafe_allow_html=True)
            cs, cb = cusum_series(resid)
            cssq, cbs = cusumsq_series(resid)
            t_ax = list(range(len(cs)))
 
            fig_cusum = make_subplots(1,2,subplot_titles=["CUSUM","CUSUM-SQ"])
            fig_cusum.add_trace(go.Scatter(x=t_ax,y=cs,mode="lines",
                                            line=dict(color=COLORS[0],width=2),name="CUSUM"),1,1)
            fig_cusum.add_trace(go.Scatter(x=t_ax,y=cb,mode="lines",
                                            line=dict(color=COLORS[3],dash="dash",width=1.5),name="+5%"),1,1)
            fig_cusum.add_trace(go.Scatter(x=t_ax,y=-cb,mode="lines",
                                            line=dict(color=COLORS[3],dash="dash",width=1.5),name="−5%"),1,1)
            fig_cusum.add_trace(go.Scatter(x=t_ax,y=cssq,mode="lines",
                                            line=dict(color=COLORS[1],width=2),name="CUSUM-SQ"),1,2)
            fig_cusum.add_trace(go.Scatter(x=t_ax,y=cbs,mode="lines",
                                            line=dict(color=COLORS[3],dash="dash",width=1.5),showlegend=False),1,2)
            fig_cusum.update_layout(**plo(T,height=280))
            for ann in fig_cusum.layout.annotations: ann.font.color=T["text"]
            st.plotly_chart(fig_cusum, use_container_width=True)
 
            # VIF (OLS only)
            if diag_model=="OLS" and has_ols:
                st.markdown(sec("Multicollinearity — VIF"), unsafe_allow_html=True)
                try:
                    exog = res.model.exog
                    vif_df = pd.DataFrame({
                        "Variable": res.model.exog_names,
                        "VIF": [variance_inflation_factor(exog,i) for i in range(exog.shape[1])],
                    })
                    vif_df["Status"] = vif_df["VIF"].apply(
                        lambda v: "🔴 High (>10)" if v>10 else "🟡 Moderate (5-10)" if v>5 else "✅ OK (<5)"
                    )
                    st.dataframe(vif_df.style.format({"VIF":"{:.2f}"}),
                                 use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"VIF could not be computed: {e}")
 
            # Export diagnostics
            d_exp = {"DW":dw,"LjungBox_p10":lb_p,"BreuschPagan_p":bp_p,
                     "ARCH_LM_p":arch_p,"JarqueBera_p":jb_p,
                     "R2":res.rsquared,"AdjR2":res.rsquared_adj}
            st.download_button("⬇ Export Diagnostics CSV",
                               results_csv(d_exp),"diagnostics.csv","text/csv")
 
        elif diag_model=="VAR" and has_var:
            res = st.session_state["var_res"]
            st.markdown(sec("VAR Stability — Eigenvalue Analysis"), unsafe_allow_html=True)
 
            roots = res.roots
            is_stable = all(np.abs(r)<1 for r in roots)
            cls = "verdict-stationary" if is_stable else "verdict-danger"
            msg = "✅ VAR is STABLE — all moduli < 1" if is_stable else "⚠️ UNSTABLE — some moduli ≥ 1"
            st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
 
            theta = np.linspace(0,2*np.pi,300)
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=np.cos(theta),y=np.sin(theta),mode="lines",
                                        line=dict(color=T["text3"],width=1),name="Unit circle"))
            fig_e.add_trace(go.Scatter(x=roots.real,y=roots.imag,mode="markers",
                                        marker=dict(color=COLORS[0],size=12,symbol="x-thin-open",
                                                    line=dict(width=2,color=COLORS[0])),
                                        name="Eigenvalues"))
            fig_e.update_layout(**plo(T,height=380,title="VAR Eigenvalue Plot"),
                                 yaxis=dict(scaleanchor="x",scaleratio=1,gridcolor=T["grid"]),
                                 xaxis_title="Real",yaxis_title="Imaginary")
            st.plotly_chart(fig_e, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════
# TAB 5 — ARIMA FORECAST
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(sec("ARIMA / Auto-ARIMA Forecasting Engine"), unsafe_allow_html=True)
    st.markdown('<div class="latex-block">', unsafe_allow_html=True)
    st.latex(r"\Phi(B)(1-B)^d y_t = c + \Theta(B)\varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma^2)")
    st.markdown("</div>", unsafe_allow_html=True)
 
    left, right = st.columns([1,2])
    with left:
        st.markdown(f'<div class="mf-card accent-left">', unsafe_allow_html=True)
        fc_var = st.selectbox("Variable to Forecast", numeric_cols)
        fc_mode = st.radio("Order Selection",
                            ["Auto (AIC grid search)","Manual"],
                            index=0)
        if fc_mode.startswith("Manual"):
            fc_p = st.number_input("p (AR)",0,6,1)
            fc_d = st.number_input("d (I)",0,2,1)
            fc_q = st.number_input("q (MA)",0,6,0)
        else:
            fc_p = fc_d = fc_q = None
        fc_h = st.number_input("Forecast Horizon (periods)", 2, 30, 8)
        run_fc = st.button("▶ Run Forecast")
        st.markdown("</div>", unsafe_allow_html=True)
 
    if run_fc:
        series_fc = pd.to_numeric(df_raw[fc_var], errors="coerce").dropna()
 
        with right:
            with st.spinner("Selecting optimal ARIMA order…" if fc_mode.startswith("Auto") else "Fitting ARIMA…"):
                try:
                    if fc_mode.startswith("Auto"):
                        best_order, best_aic = auto_arima_aic(series_fc, max_p=4, max_q=4, max_d=2)
                        p,d,q = best_order
                    else:
                        p,d,q = int(fc_p),int(fc_d),int(fc_q)
                        best_aic = ARIMA(series_fc,order=(p,d,q)).fit().aic
 
                    model_fc = ARIMA(series_fc, order=(p,d,q)).fit()
                    forecast = model_fc.get_forecast(steps=fc_h)
                    fc_mean  = forecast.predicted_mean
                    fc_ci    = forecast.conf_int(alpha=0.05)
 
                    # Build x-axis
                    hist_x = list(range(len(series_fc)))
                    fc_x   = list(range(len(series_fc), len(series_fc)+fc_h))
 
                    # Order badge
                    st.markdown(f"""
                    <div style="display:flex;gap:1rem;margin-bottom:1rem">
                      <div class="mf-card accent-left" style="flex:1;margin-bottom:0">
                        <div class="kpi-label">Selected Order</div>
                        <div class="kpi-value">ARIMA({p},{d},{q})</div>
                      </div>
                      <div class="mf-card accent-gold" style="flex:1;margin-bottom:0">
                        <div class="kpi-label">AIC</div>
                        <div class="kpi-value">{best_aic:.2f}</div>
                      </div>
                      <div class="mf-card" style="flex:1;margin-bottom:0">
                        <div class="kpi-label">Horizon</div>
                        <div class="kpi-value">{fc_h} periods</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
 
                    # Forecast chart
                    fig_fc = go.Figure()
 
                    # Historical
                    fig_fc.add_trace(go.Scatter(
                        x=hist_x, y=series_fc.values, name="Historical",
                        line=dict(color=COLORS[0],width=2)))
 
                    # In-sample fitted
                    fig_fc.add_trace(go.Scatter(
                        x=hist_x, y=model_fc.fittedvalues.values, name="In-sample fit",
                        line=dict(color=COLORS[1],width=1.5,dash="dot"),opacity=0.7))
 
                    # 95% CI shading
                    ci_upper = fc_ci.iloc[:,1].values
                    ci_lower = fc_ci.iloc[:,0].values
                    fig_fc.add_trace(go.Scatter(
                        x=fc_x+fc_x[::-1],
                        y=list(ci_upper)+list(ci_lower[::-1]),
                        fill="toself",
                        fillcolor=f"rgba(59,158,255,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="95% CI",hoverinfo="skip"))
 
                    # Forecast line
                    fig_fc.add_trace(go.Scatter(
                        x=fc_x, y=fc_mean.values, name="Forecast",
                        line=dict(color=COLORS[1],width=2.5,dash="dash"),
                        mode="lines+markers",
                        marker=dict(size=7,color=COLORS[1],
                                    line=dict(width=2,color=T["bg"]))))
 
                    # Vertical divider
                    fig_fc.add_vline(x=len(series_fc)-1,
                                      line=dict(color=T["text3"],width=1,dash="dot"))
 
                    fig_fc.update_layout(**plo(T, height=420,
                                               title=f"ARIMA({p},{d},{q}) Forecast — {fc_var}"))
                    st.plotly_chart(fig_fc, use_container_width=True)
 
                    # Forecast table
                    fc_df = pd.DataFrame({
                        "Period": [f"+{i+1}" for i in range(fc_h)],
                        "Forecast": fc_mean.values.round(4),
                        "CI Lower (95%)": ci_lower.round(4),
                        "CI Upper (95%)": ci_upper.round(4),
                    })
                    st.dataframe(fc_df, use_container_width=True, hide_index=True)
 
                    # Residual diagnostics for ARIMA
                    arima_resid = model_fc.resid.values
                    _, jb_p_a = sp_stats.jarque_bera(arima_resid)
                    arch_s_a, arch_p_a = arch_lm(arima_resid)
                    lb_a = acorr_ljungbox(arima_resid,lags=[10],return_df=True)
                    lb_p_a = lb_a["lb_pvalue"].values[0]
 
                    st.markdown(sec("ARIMA Residual Checks"), unsafe_allow_html=True)
                    ra1,ra2,ra3 = st.columns(3)
                    with ra1: st.markdown(kpi_card("Ljung-Box p",f"{lb_p_a:.4f}","Lag 10",
                                                    accent="accent-left" if lb_p_a>sig_level else "accent-danger"), unsafe_allow_html=True)
                    with ra2: st.markdown(kpi_card("ARCH-LM p","—" if np.isnan(arch_p_a) else f"{arch_p_a:.4f}",
                                                    "Volatility clustering",
                                                    accent="accent-left" if (np.isnan(arch_p_a) or arch_p_a>sig_level) else "accent-gold"), unsafe_allow_html=True)
                    with ra3: st.markdown(kpi_card("Jarque-Bera p",f"{jb_p_a:.4f}","Normality",
                                                    accent="accent-left" if jb_p_a>sig_level else "accent-gold"), unsafe_allow_html=True)
 
                    # Export
                    export_fc = {
                        "Variable": fc_var,
                        "ARIMA_order": f"({p},{d},{q})",
                        "AIC": best_aic,
                        **{f"Forecast_t+{i+1}": v for i,v in enumerate(fc_mean.values)},
                        **{f"CI_lower_t+{i+1}": v for i,v in enumerate(ci_lower)},
                        **{f"CI_upper_t+{i+1}": v for i,v in enumerate(ci_upper)},
                    }
                    c_ex1, c_ex2 = st.columns(2)
                    with c_ex1:
                        st.download_button("⬇ Export Forecast CSV",
                                           results_csv(export_fc),"arima_forecast.csv","text/csv")
                    with c_ex2:
                        summary_body = (f"Variable: {fc_var}\nOrder: ARIMA({p},{d},{q})\n"
                                        f"AIC: {best_aic:.4f}\nHorizon: {fc_h}\n\n"
                                        + fc_df.to_string(index=False))
                        st.download_button("⬇ Export Summary TXT",
                                           results_txt("ARIMA Forecast",summary_body),
                                           "arima_summary.txt")
 
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")
 
