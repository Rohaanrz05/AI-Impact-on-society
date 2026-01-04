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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Nexus | Ultra Dashboard", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- 🎨 ULTRA CSS THEME (Pretty & Clean) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --bg-dark: #0a0e17;
        --card-bg: rgba(30, 41, 59, 0.4);
        --text-highlight: #38bdf8;
    }

    /* APP BACKGROUND */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), 
            radial-gradient(at 90% 10%, rgba(236, 72, 153, 0.15) 0px, transparent 50%), 
            radial-gradient(at 50% 90%, rgba(16, 185, 129, 0.10) 0px, transparent 50%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.5); border-radius: 10px; }

    /* 🔮 GLASS CARDS (Container Design) */
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

    /* HEADERS */
    h1, h2, h3 { color: white !important; font-family: 'Outfit', sans-serif !important; }
    
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
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
        st.markdown("<h2 style='text-align: center;'>🧬 AI NEXUS</h2>", unsafe_allow_html=True)
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
        st.markdown("<h1 class='gradient-text'>🛸 COMMAND CENTER</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Real-time AI perception analytics.</p>", unsafe_allow_html=True)
        
        # --- IMPROVED KPI ROW ---
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
            ("Job Concerns", f"{job_concern_count} Users", "Said 'Yes' to Job Loss") # FIXED: Shows Count
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
        
        # --- SEPARATE GRAPH DESIGNS (2D Only) ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Demographics: Age")
            # 2D Histogram
            fig_age = px.histogram(df, x='Age Range', color='Age Range', 
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_age.update_layout(height=350, margin=dict(t=30, l=0, r=0, b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False, font_color='white')
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            # 2D Donut Chart
            fig_trust = px.pie(df, names='Trust in AI', hole=0.5, 
                               color_discrete_sequence=px.colors.sequential.Bluyl)
            fig_trust.update_layout(height=350, margin=dict(t=30, l=0, r=0, b=0), 
                                    paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_trust, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 2: PREDICTION LAB (Perfect ML Mode) ---
    elif page == "🔮 Prediction Lab":
        st.markdown("<h1 class='gradient-text'>🔮 PREDICTION LAB</h1>", unsafe_allow_html=True)
        
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
                
                st.info(f"⚡ Predictions generated using **{m_name}** algorithm based on the provided profile.")
            
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 50px;">
                    <h3>Ready to Predict</h3>
                    <p style="color: #94a3b8;">Select parameters on the left and click 'Generate Prediction' to see the dual-model AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)

    # --- PAGE 3: DEEP MATRIX (2D Charts Only) ---
    elif page == "🧠 Deep Matrix":
        st.markdown("<h1 class='gradient-text'>🧠 DEEP MATRIX</h1>", unsafe_allow_html=True)
        
        # 2D Bubble Chart (Replacing 3D)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🫧 Age vs Usage Patterns (by Trust)")
        fig_bubble = px.scatter(
            df, x='Age Range', y='AI Usage Rating',
            color='Trust in AI', size='AI Usage Rating', # Bubble size based on usage
            symbol='Gender',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['Education Level']
        )
        fig_bubble.update_layout(
            height=500, margin=dict(t=30, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Heatmap (2D)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Variable Correlation Heatmap")
        d_corr = df.apply(lambda x: pd.factorize(x)[0])
        corr = d_corr.corr()
        fig_h = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Magma', text=corr.values, texttemplate='%{text:.1f}'
        ))
        fig_h.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 4: MODEL ARENA ---
    elif page == "⚡ Model Arena":
        st.markdown("<h1 class='gradient-text'>⚡ MODEL ARENA</h1>", unsafe_allow_html=True)
        
        target = st.selectbox("Select Target Variable", ["Employment Status", "Trust in AI"])
        
        # Quick Train for Comparison
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
            
        # Comparison Bar Chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Performance Comparison")
        comp_df = pd.DataFrame({'Model': ['Random Forest', 'XGBoost'], 'Accuracy': [acc_rf, acc_xgb]})
        fig_comp = px.bar(comp_df, x='Model', y='Accuracy', color='Model', 
                          color_discrete_sequence=['#10b981', '#3b82f6'], text_auto='.1%')
        fig_comp.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("🚨 CRITICAL ERROR: Database Connection Failed. Please ensure 'cleaned_ai_impact_data_updated.csv' is in the root directory.")
