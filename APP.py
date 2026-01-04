import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Nexus Dashboard", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (SAFE MODE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.6);
        transform: scale(1.02);
    }

    h1, h2, h3 { color: white !important; }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'cleaned_ai_impact_data_updated.csv'
    for enc in ['utf-8', 'ISO-8859-1', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df.columns = df.columns.str.strip()
            
            rename_map = {
                'Age_Range': 'Age Range', 'Employment_Status': 'Employment Status',
                'AI_Knowledge': 'AI Knowledge', 'AI_Trust': 'Trust in AI',
                'AI_Usage_Scale': 'AI Usage Rating', 'Education': 'Education Level',
                'Future_AI_Usage': 'Future AI Interest', 'Eliminate_Jobs': 'AI Job Impact',
                'Threaten_Freedoms': 'AI Impact Perception'
            }
            df.rename(columns=rename_map, inplace=True)
            
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
                
            return df
        except:
            continue
    return None

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def build_models(df, target_col, feature_cols):
    ml_df = df[feature_cols + [target_col]].dropna()
    encoders = {}
    
    # Fit Encoders
    for col in ml_df.columns:
        le = LabelEncoder()
        # Convert to string to ensure consistency
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le
        
    X = ml_df[feature_cols]
    y = ml_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
    
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    
    return rf, xgb, acc_rf, acc_xgb, encoders

# --- HELPER: SAFE ENCODE FUNCTION ---
def safe_encode(le, value):
    """Safely transform value; if unseen, default to first class."""
    try:
        # Force string conversion to match training
        return le.transform([str(value)])[0]
    except ValueError:
        # Fallback for unseen labels
        return 0 

# --- 5. MAIN APP LOGIC ---
df = load_data()

if df is not None:
    
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #8b5cf6;'>🚀 AI NEXUS</h2>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("MAIN MENU", ["📊 Dashboard", "🔮 Prediction Lab", "⚡ Model Arena"])
        st.markdown("---")
        
        # Global Filter
        st.markdown("### ⚙️ Global Filter")
        all_genders = df['Gender'].unique().tolist()
        sel_gender = st.multiselect("Filter by Gender", all_genders, default=all_genders)
        
        if sel_gender:
            df = df[df['Gender'].isin(sel_gender)]
            
        st.caption(f"Showing {len(df)} records")

    # --- TAB 1: DASHBOARD ---
    if menu == "📊 Dashboard":
        st.markdown("<div class='hero-title'>COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94a3b8;'>Real-time analysis of AI perception</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if 'AI Job Impact' in df.columns:
            job_count = len(df[df['AI Job Impact'].str.lower() == 'yes'])
        else:
            job_count = 0

        c1, c2, c3, c4 = st.columns(4)
        
        def card(col, label, value, sub):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size:0.8rem; color:#64748b;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        card(c1, "Total Records", len(df), "Active Data Points")
        card(c2, "Trust Rate", f"{(df['Trust in AI'].str.contains('trust', case=False).mean()*100):.0f}%", "Users Trust AI")
        card(c3, "Avg Usage", f"{df['AI Usage Rating'].astype(float).mean():.1f}/5", "Usage Intensity")
        card(c4, "Job Concerns", f"{job_count} People", "Said 'Yes' to Job Loss")

        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Age Demographics")
            fig1 = px.histogram(df, x='Age Range', color='Age Range', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            fig2 = px.pie(df, names='Trust in AI', hole=0.5, color_discrete_sequence=px.colors.sequential.Bluyl)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: PREDICTION LAB ---
    elif menu == "🔮 Prediction Lab":
        st.markdown("<div class='hero-title'>PREDICTION LAB</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. SETUP ML MODELS
        feats_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        rf_e, xgb_e, acc_e, acc_xgb_e, enc_e = build_models(df, 'Employment Status', feats_emp)
        
        feats_tru = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
        rf_t, xgb_t, acc_t, acc_xgb_t, enc_t = build_models(df, 'Trust in AI', feats_tru)
        
        # 2. USER INPUTS
        c_in, c_out = st.columns([1, 1.5])
        
        with c_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👤 Configure Profile")
            
            model_engine = st.radio("Select Engine", ["⚡ XGBoost", "🌲 Random Forest"])
            
            # Select boxes use classes from encoder
            in_age = st.selectbox("Age", enc_e['Age Range'].classes_)
            in_gen = st.selectbox("Gender", enc_e['Gender'].classes_)
            in_edu = st.selectbox("Education", enc_e['Education Level'].classes_)
            in_use = st.slider("Usage (1-5)", 1, 5, 3)
            in_know = st.select_slider("Knowledge", options=enc_t['AI Knowledge'].classes_)
            
            run_pred = st.button("🚀 PREDICT NOW")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 3. RESULTS
        with c_out:
            if run_pred:
                # --- FIX: USE SAFE_ENCODE ---
                input_e = [safe_encode(enc_e[c], val) for c, val in zip(feats_emp, [in_age, in_gen, in_edu, in_use])]
                input_t = [safe_encode(enc_t[c], val) for c, val in zip(feats_tru, [in_age, in_gen, in_edu, in_use, in_know])]
                
                # Logic
                if "XGBoost" in model_engine:
                    pred_emp = enc_e['Employment Status'].inverse_transform(xgb_e.predict([input_e]))[0]
                    conf_emp = acc_xgb_e
                    pred_tru = enc_t['Trust in AI'].inverse_transform(xgb_t.predict([input_t]))[0]
                    conf_tru = acc_xgb_t
                else:
                    pred_emp = enc_e['Employment Status'].inverse_transform(rf_e.predict([input_e]))[0]
                    conf_emp = acc_e
                    pred_tru = enc_t['Trust in AI'].inverse_transform(rf_t.predict([input_t]))[0]
                    conf_tru = acc_t
                
                # Display Results
                st.markdown(f"""
                <div style="display:flex; gap:20px; margin-bottom:20px;">
                    <div class="glass-card" style="flex:1; text-align:center; border-left: 4px solid #10b981;">
                        <h4 style="color:#10b981; margin:0;">💼 Employment</h4>
                        <div class="metric-value" style="font-size:2rem; margin:10px 0;">{pred_emp}</div>
                        <div style="color:#94a3b8;">Accuracy: {conf_emp:.1%}</div>
                    </div>
                    <div class="glass-card" style="flex:1; text-align:center; border-left: 4px solid #ec4899;">
                        <h4 style="color:#ec4899; margin:0;">🤝 Trust Level</h4>
                        <div class="metric-value" style="font-size:2rem; margin:10px 0;">{pred_tru}</div>
                        <div style="color:#94a3b8;">Accuracy: {conf_tru:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success("Analysis Complete!")
            
            else:
                st.info("👈 Select options and click 'Predict Now' to see the dual-model results.")

    # --- TAB 3: MODEL ARENA ---
    elif menu == "⚡ Model Arena":
        st.markdown("<div class='hero-title'>MODEL ARENA</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        target_select = st.selectbox("Compare Performance For:", ["Employment Status", "Trust in AI"])
        
        if target_select == "Employment Status":
            fts = feats_emp
        else:
            fts = feats_tru
            
        rf, xgb, rf_sc, xgb_sc, _ = build_models(df, target_select, fts)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; border-top: 4px solid #10b981;">
                <h3>🌲 Random Forest</h3>
                <div class="metric-value" style="color:#10b981;">{rf_sc:.1%}</div>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; border-top: 4px solid #3b82f6;">
                <h3>⚡ XGBoost</h3>
                <div class="metric-value" style="color:#3b82f6;">{xgb_sc:.1%}</div>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Algorithm Showdown")
        res_df = pd.DataFrame({'Model': ['Random Forest', 'XGBoost'], 'Accuracy': [rf_sc, xgb_sc]})
        fig_comp = px.bar(res_df, x='Model', y='Accuracy', color='Model', 
                          color_discrete_sequence=['#10b981', '#3b82f6'], text_auto='.1%')
        fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                               font_color='white', height=300)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("🚨 File Not Found: Please upload 'cleaned_ai_impact_data_updated.csv' to the folder.")import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Nexus Dashboard", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- ULTRA PREMIUM DARK THEME CSS (FIXED Z-INDEX) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --bg-dark: #0a0e17;
        --card-bg: rgba(30, 41, 59, 0.4);
    }

    /* 1. FIX BLANK SCREEN: Set z-index to -1 so background stays BEHIND content */
    .stApp {
        background-color: var(--bg-dark);
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), 
            radial-gradient(at 90% 10%, rgba(236, 72, 153, 0.15) 0px, transparent 50%), 
            radial-gradient(at 50% 90%, rgba(16, 185, 129, 0.10) 0px, transparent 50%);
        background-size: 200% 200%;
        animation: gradientAnimation 20s ease infinite;
        z-index: -1; /* CRITICAL FIX */
        pointer-events: none;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.5); border-radius: 10px; }

    /* 🔮 GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 15px 30px -10px rgba(99, 102, 241, 0.3);
    }

    /* METRIC TEXT */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }

    /* HEADERS & TEXT VISIBILITY */
    h1, h2, h3, p, span, div { 
        color: #e2e8f0; 
        font-family: 'Outfit', sans-serif;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def get_data():
    file_path = 'cleaned_ai_impact_data_updated.csv'
    for enc in ['utf-8', 'ISO-8859-1', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df.columns = df.columns.str.strip()
            rename_map = {
                'Age_Range': 'Age Range', 'Employment_Status': 'Employment Status',
                'AI_Knowledge': 'AI Knowledge', 'AI_Trust': 'Trust in AI',
                'AI_Usage_Scale': 'AI Usage Rating', 'Education': 'Education Level',
                'Future_AI_Usage': 'Future AI Interest', 'Eliminate_Jobs': 'AI Job Impact',
                'Threaten_Freedoms': 'AI Impact Perception'
            }
            df.rename(columns=rename_map, inplace=True)
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
            return df
        except:
            continue
    return None

# --- ML ENGINE ---
@st.cache_resource
def build_model(df, target_col, features):
    ml_df = df[features + [target_col]].dropna()
    encoders = {}
    for col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le
    X = ml_df[features]
    y = ml_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    return rf, xgb, acc_rf, acc_xgb, encoders

df_raw = get_data()

# --- APP LAYOUT ---
if df_raw is not None:
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #8b5cf6;'>🧬 AI NEXUS</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        page = st.radio("INTERFACE", [
            "🛸 Command Center", 
            "🔮 Prediction Lab", 
            "🧠 Deep Matrix", 
            "⚡ Model Arena"
        ])
        
        st.markdown("---")
        st.markdown("### 🎛️ Global Filters")
        filter_gender = st.multiselect("Gender", df_raw['Gender'].unique(), default=df_raw['Gender'].unique())
        
        df = df_raw[df_raw['Gender'].isin(filter_gender)]
        st.markdown(f"<div style='text-align:center; color:#64748b; margin-top:10px;'>Active Records: {len(df)}</div>", unsafe_allow_html=True)

    # --- PAGE 1: COMMAND CENTER (Dashboard) ---
    if page == "🛸 Command Center":
        st.markdown("<div class='gradient-text'>🛸 COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Real-time AI perception analytics.</p>", unsafe_allow_html=True)
        st.write("") # Spacer
        
        # Calculate Job Concern Count
        if 'AI Job Impact' in df.columns:
            job_concern_count = df[df['AI Job Impact'].astype(str).str.lower() == 'yes'].shape[0]
        else:
            job_concern_count = 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        metrics = [
            ("Total Records", len(df), "Filtered View"),
            ("Avg Trust", f"{(df['Trust in AI'].str.contains('trust', case=False).mean()*100):.1f}%", "Trusting Users"),
            ("Usage Intensity", f"{df['AI Usage Rating'].astype(float).mean():.1f}/5", "Activity Level"),
            ("Job Concerns", f"{job_concern_count} Users", "Said 'Yes' to Job Loss") 
        ]
        
        for col, (label, val, sub) in zip([kpi1, kpi2, kpi3, kpi4], metrics):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 20px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div style="color: #64748b; font-size: 0.8rem; margin-top: 5px;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # --- GRAPH DESIGNS (2D) ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Demographics: Age")
            fig_age = px.histogram(df, x='Age Range', color='Age Range', 
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_age.update_layout(height=350, margin=dict(t=30, l=0, r=0, b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False, font=dict(color='white'))
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            fig_trust = px.pie(df, names='Trust in AI', hole=0.5, 
                               color_discrete_sequence=px.colors.sequential.Bluyl)
            fig_trust.update_layout(height=350, margin=dict(t=30, l=0, r=0, b=0), 
                                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_trust, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 2: PREDICTION LAB ---
    elif page == "🔮 Prediction Lab":
        st.markdown("<div class='gradient-text'>🔮 PREDICTION LAB</div>", unsafe_allow_html=True)
        st.write("")
        
        # Configuration
        col_input, col_result = st.columns([1, 1.5])
        
        with col_input:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👤 User Profile")
            
            # Setup Models
            feats_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
            rf_e, xgb_e, acc_e, acc_xgb_e, enc_e = build_model(df_raw, 'Employment Status', feats_emp)
            
            feats_tru = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
            rf_t, xgb_t, acc_t, acc_xgb_t, enc_t = build_model(df_raw, 'Trust in AI', feats_tru)
            
            # Model Selection
            model_type = st.radio("Select Model Engine", ["⚡ XGBoost (Recommended)", "🌲 Random Forest"])
            
            # Inputs
            u_age = st.selectbox("Age Range", enc_e['Age Range'].classes_)
            u_gen = st.selectbox("Gender", enc_e['Gender'].classes_)
            u_edu = st.selectbox("Education", enc_e['Education Level'].classes_)
            u_use = st.slider("Usage Level (1-5)", 1, 5, 3)
            u_know = st.select_slider("Knowledge Level", options=enc_t['AI Knowledge'].classes_)
            
            btn_predict = st.button("🚀 GENERATE PREDICTION")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            if btn_predict:
                # Prepare Inputs
                in_e = [enc_e[c].transform([val])[0] for c, val in zip(feats_emp, [u_age, u_gen, u_edu, u_use])]
                in_t = [enc_t[c].transform([val])[0] for c, val in zip(feats_tru, [u_age, u_gen, u_edu, u_use, u_know])]
                
                # Logic to pick model
                if "XGBoost" in model_type:
                    pred_emp = enc_e['Employment Status'].inverse_transform(xgb_e.predict([in_e]))[0]
                    conf_emp = acc_xgb_e
                    pred_tru = enc_t['Trust in AI'].inverse_transform(xgb_t.predict([in_t]))[0]
                    conf_tru = acc_xgb_t
                    m_name = "XGBoost"
                else:
                    pred_emp = enc_e['Employment Status'].inverse_transform(rf_e.predict([in_e]))[0]
                    conf_emp = acc_e
                    pred_tru = enc_t['Trust in AI'].inverse_transform(rf_t.predict([in_t]))[0]
                    conf_tru = acc_t
                    m_name = "Random Forest"

                # Dual Result Display
                st.markdown(f"""
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                    <div class="glass-card" style="flex: 1; border-color: #10b981; text-align: center;">
                        <h4 style="color: #10b981; margin: 0;">💼 Employment Status</h4>
                        <div class="metric-value" style="font-size: 2.2rem; margin: 10px 0;">{pred_emp}</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Confidence: {conf_emp:.1%}</div>
                    </div>
                    <div class="glass-card" style="flex: 1; border-color: #ec4899; text-align: center;">
                        <h4 style="color: #ec4899; margin: 0;">🤝 Trust Level</h4>
                        <div class="metric-value" style="font-size: 2.2rem; margin: 10px 0;">{pred_tru}</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Confidence: {conf_tru:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"⚡ Predictions generated using **{m_name}**.")
            
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 50px;">
                    <h3>Ready to Predict</h3>
                    <p style="color: #94a3b8;">Select parameters on the left and click 'Generate Prediction'.</p>
                </div>
                """, unsafe_allow_html=True)

    # --- PAGE 3: DEEP MATRIX ---
    elif page == "🧠 Deep Matrix":
        st.markdown("<div class='gradient-text'>🧠 DEEP MATRIX</div>", unsafe_allow_html=True)
        st.write("")
        
        # 2D Bubble Chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🫧 Age vs Usage Patterns (by Trust)")
        fig_bubble = px.scatter(
            df, x='Age Range', y='AI Usage Rating',
            color='Trust in AI', size='AI Usage Rating',
            symbol='Gender',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['Education Level']
        )
        fig_bubble.update_layout(
            height=500, margin=dict(t=30, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Heatmap
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Variable Correlation Heatmap")
        d_corr = df.apply(lambda x: pd.factorize(x)[0])
        corr = d_corr.corr()
        fig_h = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Magma', text=corr.values, texttemplate='%{text:.1f}'
        ))
        fig_h.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 4: MODEL ARENA ---
    elif page == "⚡ Model Arena":
        st.markdown("<div class='gradient-text'>⚡ MODEL ARENA</div>", unsafe_allow_html=True)
        st.write("")
        
        target = st.selectbox("Select Target Variable", ["Employment Status", "Trust in AI"])
        
        feats = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge'] if target == 'Trust in AI' else ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        rf, xgb, acc_rf, acc_xgb, _ = build_model(df_raw, target, feats)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid #10b981; text-align: center;">
                <h3>🌲 Random Forest</h3>
                <div class="metric-value" style="color: #10b981 !important;">{acc_rf:.1%}</div>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid #3b82f6; text-align: center;">
                <h3>⚡ XGBoost</h3>
                <div class="metric-value" style="color: #3b82f6 !important;">{acc_xgb:.1%}</div>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Performance Comparison")
        comp_df = pd.DataFrame({'Model': ['Random Forest', 'XGBoost'], 'Accuracy': [acc_rf, acc_xgb]})
        fig_comp = px.bar(comp_df, x='Model', y='Accuracy', color='Model', 
                          color_discrete_sequence=['#10b981', '#3b82f6'], text_auto='.1%')
        fig_comp.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("🚨 CRITICAL ERROR: Database Connection Failed. Please ensure 'cleaned_ai_impact_data_updated.csv' is in the root directory.")

