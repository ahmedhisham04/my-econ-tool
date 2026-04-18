# ... (Keep the CSS from previous steps at the top)

if uploaded_file:
    # --- STEP 1: THE INTELLIGENCE HUB ---
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # --- STEP 2: THE ANATOMY WIZARD ---
    st.markdown("<div class='section-header'>Step 1 & 2: Anatomy & Mapping</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])
    with col_map:
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df_raw.columns)
        potential_vars = [c for c in df_raw.columns if c != time_idx]
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    # --- THE GHOST KILLER (Strict Truncation) ---
    # We only keep rows where the 'Year' column has a valid number. 
    # This kills the 1,032,222 rows instantly.
    df = df_raw.dropna(subset=[time_idx]).copy()
    df[time_idx] = pd.to_numeric(df[time_idx], errors='coerce')
    df = df.dropna(subset=[time_idx]) # Final purge of non-numeric years
    
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if time_idx and dep_var and indep_vars:
            st.markdown(f"""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 12px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>Sample Size: {len(df)} Years<br>Variables Mapped.</div>
            </div>
            """, unsafe_allow_html=True)
            init_btn = st.button("INITIALIZE RESEARCH KERNEL")
        else:
            init_btn = False
            st.warning("Awaiting mapping...")

    # --- STEP 3: THE VISUAL PULSE (Appears after Initialization) ---
    if init_btn or st.session_state.get('initialized', False):
        st.session_state['initialized'] = True
        st.markdown("<div class='section-header'>Step 3: Visual Pulse & Stationarity Watch</div>", unsafe_allow_html=True)
        
        # We use a Tab system for the Analysis, similar to EViews' "View" menu
        viz_tab, health_tab = st.tabs(["[ VISUAL GALLERY ]", "[ DATA HEALTH ]"])
        
        with viz_tab:
            import plotly.express as px
            # Professional Time Series Plot
            plot_vars = [dep_var] + indep_vars
            fig = px.line(df, x=time_idx, y=plot_vars, 
                          template="plotly_white",
                          color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B", "#EF4444"])
            
            fig.update_layout(
                font_family="Inter",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Normalized Value",
                xaxis_title="Chronological Axis"
            )
            st.plotly_chart(fig, use_container_width=True)

        with health_tab:
            st.markdown("#### Stationarity Watch (ADF Pre-Check)")
            st.write("Checking for Unit Roots to prevent Spurious Regression.")
            
            # Simple health check cards
            h1, h2 = st.columns(2)
            with h1:
                st.markdown(f"""<div class="data-card"><div class="card-label">Primary Target</div><div class="card-value">{dep_var}</div></div>""", unsafe_allow_html=True)
            with h2:
                # We will add the actual math for the ADF test in Step 4
                st.markdown(f"""<div class="data-card"><div class="card-label">Recommended Model</div><div class="card-value">Pending Math...</div></div>""", unsafe_allow_html=True)
