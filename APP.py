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

# --- 2. CSS STYLING (SAFE MODE - NO BLANK SCREEN) ---
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply Font Globally */
    * { font-family: 'Outfit', sans-serif; }

    /* BACKGROUND - Safe Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
    }

    /* GLASS CARDS (Containers) */
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

    /* KPI METRICS */
    .metric-container {
        text-align: center;
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

    /* BUTTONS */
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

    /* TITLES */
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
    
    /* REMOVE DEFAULT STREAMLIT PADDING */
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'cleaned_ai_impact_data_updated.csv'
    # Try different encodings to ensure it loads
    for enc in ['utf-8', 'ISO-8859-1', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Renaming for consistency
            rename_map = {
                'Age_Range': 'Age Range', 
                'Employment_Status': 'Employment Status',
                'AI_Knowledge': 'AI Knowledge', 
                'AI_Trust': 'Trust in AI',
                'AI_Usage_Scale': 'AI Usage Rating', 
                'Education': 'Education Level',
                'Future_AI_Usage': 'Future AI Interest', 
                'Eliminate_Jobs': 'AI Job Impact',
                'Threaten_Freedoms': 'AI Impact Perception'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Clean string values
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
                
            return df
        except:
            continue
    return None

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def build_models(df, target_col, feature_cols):
    # 1. Prepare Data
    ml_df = df[feature_cols + [target_col]].dropna()
    
    # 2. Encode Strings to Numbers
    encoders = {}
    for col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le
        
    X = ml_df[feature_cols]
    y = ml_df[target_col]
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Models
    rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
    
    # 5. Score
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    
    return rf, xgb, acc_rf, acc_xgb, encoders

# --- 5. MAIN APP LOGIC ---
df = load_data()

if df is not None:
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #8b5cf6;'>🚀 AI NEXUS</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        menu = st.radio("MAIN MENU", [
            "📊 Dashboard", 
            "🔮 Prediction Lab", 
            "⚡ Model Arena"
        ])
        
        st.markdown("---")
        st.markdown("### ⚙️ Global Filter")
        # Global Filter
        all_genders = df['Gender'].unique().tolist()
        sel_gender = st.multiselect("Filter by Gender", all_genders, default=all_genders)
        
        # Apply Filter
        if sel_gender:
            df = df[df['Gender'].isin(sel_gender)]
            
        st.caption(f"Showing {len(df)} records")

    # --- TAB 1: DASHBOARD ---
    if menu == "📊 Dashboard":
        st.markdown("<div class='hero-title'>COMMAND CENTER</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94a3b8;'>Real-time analysis of AI perception</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- KPIS ---
        # Calculate Job Concern Count specifically
        if 'AI Job Impact' in df.columns:
            # Counts rows where users said "Yes" (case insensitive)
            job_count = len(df[df['AI Job Impact'].str.lower() == 'yes'])
        else:
            job_count = 0

        c1, c2, c3, c4 = st.columns(4)
        
        # Helper for HTML Card
        def card(col, label, value, sub):
            with col:
                st.markdown(f"""
                <div class="glass-card metric-container">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size:0.8rem; color:#64748b;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        card(c1, "Total Records", len(df), "Active Data Points")
        card(c2, "Trust Rate", f"{(df['Trust in AI'].str.contains('trust', case=False).mean()*100):.0f}%", "Users Trust AI")
        card(c3, "Avg Usage", f"{df['AI Usage Rating'].astype(float).mean():.1f}/5", "Usage Intensity")
        card(c4, "Job Concerns", f"{job_count} People", "Said 'Yes' to Job Loss") # <--- FIXED HERE

        # --- CHARTS ---
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Age Demographics")
            fig1 = px.histogram(df, x='Age Range', color='Age Range', 
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                               font_color='white', showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            fig2 = px.pie(df, names='Trust in AI', hole=0.5, 
                          color_discrete_sequence=px.colors.sequential.Bluyl)
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
            
            # Model Choice
            model_engine = st.radio("Select Engine", ["⚡ XGBoost", "🌲 Random Forest"])
            
            # Data Inputs
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
                # Prepare input arrays
                input_e = [enc_e[c].transform([val])[0] for c, val in zip(feats_emp, [in_age, in_gen, in_edu, in_use])]
                input_t = [enc_t[c].transform([val])[0] for c, val in zip(feats_tru, [in_age, in_gen, in_edu, in_use, in_know])]
                
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
        
        # Define Features based on target
        if target_select == "Employment Status":
            fts = feats_emp
        else:
            fts = feats_tru
            
        # Build Models on the fly for stats
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
            
        # Bar Chart Comparison
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
    st.error("🚨 File Not Found: Please upload 'cleaned_ai_impact_data_updated.csv' to the folder.")
