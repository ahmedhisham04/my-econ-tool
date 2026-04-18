# ... (Keep all your CSS and Imports at the top)

if uploaded_file:
    # --- STEP 1: AGGRESSIVE INGESTION ---
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # 1. FORCE ALL COLUMNS TO NUMERIC (The Fix for the 2 missing variables)
    # This turns "Egypt, Arab Rep." into NaN so we can clean it
    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # 2. GHOST KILLER: Identify the Year column and kill everything else
    # We look for the column with the most "Year-like" numbers
    st.markdown("<div class='section-header'>Step 1 & 2: Anatomy & Mapping</div>", unsafe_allow_html=True)
    
    col_map, col_status = st.columns([2, 1])
    with col_map:
        time_idx = st.selectbox("Assign Temporal Anchor (Year)", options=df_raw.columns)
        
        # Now that we forced numeric conversion, all 5 variables will show up here
        potential_vars = [c for c in df_raw.columns if c != time_idx]
        
        # CLEANING DATA BASED ON YEAR (This fixes the 1 million rows)
        df = df_raw.dropna(subset=[time_idx]).copy()
        df = df[df[time_idx] > 1900] # Safety: Only keep rows with real years
        df = df.sort_values(by=time_idx)
        
        dep_var = st.selectbox("Assign Target Variable (Y)", options=potential_vars)
        indep_vars = st.multiselect("Assign Regressors (X)", options=[v for v in potential_vars if v != dep_var])

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if time_idx and dep_var and indep_vars:
            st.markdown(f"""
            <div style='background: #F0FDF4; border: 1px solid #BBF7D0; padding: 1.5rem; border-radius: 4px;'>
                <div style='color: #166534; font-size: 12px; font-weight: 600; font-family: IBM Plex Mono;'>[ READY ]</div>
                <div style='color: #166534; font-size: 12px; margin-top:5px;'>Valid Range: {int(df[time_idx].min())} - {int(df[time_idx].max())}<br>Variables: {len(potential_vars)} Detected</div>
            </div>
            """, unsafe_allow_html=True)
            init_btn = st.button("INITIALIZE RESEARCH KERNEL")
        else:
            init_btn = False
            st.warning("Mapping required...")

    # --- STEP 3: THE VISUAL PULSE ---
    if init_btn or st.session_state.get('initialized', False):
        st.session_state['initialized'] = True
        st.markdown("<div class='section-header'>Step 3: Visual Pulse</div>", unsafe_allow_html=True)
        
        viz_tab, data_tab = st.tabs(["[ VISUAL GALLERY ]", "[ RAW DATA AUDIT ]"])
        
        with viz_tab:
            import plotly.express as px
            # All selected variables are now numeric and ready to plot
            plot_vars = [dep_var] + indep_vars
            fig = px.line(df, x=time_idx, y=plot_vars, 
                          template="plotly_white",
                          color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B", "#EF4444"])
            
            fig.update_layout(font_family="Inter", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        with data_tab:
            # This allows you to verify that the numbers are correct
            st.dataframe(df[[time_idx] + plot_vars].style.format(precision=4), use_container_width=True)

else:
    st.markdown("<div style='text-align:center; padding-top:100px; color:#94A3B8'>AWAITING DATASET</div>", unsafe_allow_html=True)
