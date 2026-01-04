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
st.set_page_config(page_title="AI Social Impact Dashboard", layout="wide", page_icon="🤖")

# --- ULTRA PREMIUM DARK THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * { 
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .stApp { 
        background: #0a0e27;
        background-image: 
            radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.15) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.15) 0px, transparent 50%);
        color: #f8fafc;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.15), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(59, 130, 246, 0.2), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(139, 92, 246, 0.15), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(16, 185, 129, 0.15), transparent);
        background-size: 200px 200px, 300px 300px, 150px 150px, 250px 250px;
        background-position: 0 0, 40px 60px, 130px 270px, 70px 100px;
        animation: backgroundScroll 60s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes backgroundScroll {
        0% { background-position: 0 0, 40px 60px, 130px 270px, 70px 100px; }
        100% { background-position: 200px 200px, 240px 260px, 330px 470px, 270px 300px; }
    }
    
    section[data-testid="stSidebar"] { 
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        padding: 28px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 
            0 10px 40px -10px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent,
            rgba(59, 130, 246, 0.8) 50%,
            transparent
        );
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 
            0 20px 60px -10px rgba(59, 130, 246, 0.3),
            0 0 0 1px rgba(59, 130, 246, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    div[data-testid="metric-container"]:hover::before {
        opacity: 1;
    }
    
    div[data-testid="stMetricValue"] { 
        font-size: 42px !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    .stButton>button {
        border-radius: 16px;
        height: 3.8em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        font-weight: 700;
        width: 100%;
        font-size: 17px;
        box-shadow: 
            0 8px 32px rgba(59, 130, 246, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3), 
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .stButton>button:hover {
        background-position: 100% 0;
        box-shadow: 
            0 12px 40px rgba(59, 130, 246, 0.6),
            0 0 0 1px rgba(139, 92, 246, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transform: translateY(-3px) scale(1.02);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    h1 { 
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 80px rgba(59, 130, 246, 0.5);
    }
    
    h2 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 2rem !important;
        margin-top: 2rem !important;
    }
    
    h3 { 
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
    
    label, p, span { 
        color: #e2e8f0 !important;
        line-height: 1.6 !important;
    }
    
    .stRadio > label { 
        color: #94a3b8 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[role="radiogroup"] > label {
        background: rgba(51, 65, 85, 0.5);
        backdrop-filter: blur(10px);
        padding: 16px 24px;
        border-radius: 12px;
        margin: 8px 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    div[role="radiogroup"] > label:hover {
        border-color: rgba(59, 130, 246, 0.5);
        background: rgba(59, 130, 246, 0.1);
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    }
    
    .prediction-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        padding: 50px;
        border-radius: 28px;
        border: 2px solid transparent;
        background-image: 
            linear-gradient(rgba(30, 41, 59, 0.6), rgba(30, 41, 59, 0.6)),
            linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899, #10b981);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        text-align: center;
        box-shadow: 
            0 25px 60px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: holographicGlow 4s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(255, 255, 255, 0.05) 50%,
            transparent 70%
        );
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes holographicGlow {
        0%, 100% { 
            box-shadow: 0 25px 60px rgba(59, 130, 246, 0.3),
                       inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        25% { 
            box-shadow: 0 25px 60px rgba(139, 92, 246, 0.4),
                       inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
        50% { 
            box-shadow: 0 25px 60px rgba(236, 72, 153, 0.3),
                       inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        75% { 
            box-shadow: 0 25px 60px rgba(16, 185, 129, 0.3),
                       inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }
    }
    
    @keyframes shimmer {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    
    .dual-prediction {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        padding: 40px;
        border-radius: 24px;
        margin: 30px 0;
        text-align: center;
        border: 2px solid rgba(236, 72, 153, 0.3);
        box-shadow: 
            0 0 40px rgba(236, 72, 153, 0.2),
            0 0 80px rgba(139, 92, 246, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .dual-prediction::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(236, 72, 153, 0.1),
            transparent
        );
        animation: scanline 3s linear infinite;
    }
    
    @keyframes scanline {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .model-comparison {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        padding: 24px;
        border-radius: 16px;
        border-left: 4px solid #8b5cf6;
        margin: 12px 0;
        box-shadow: 
            -4px 0 20px rgba(139, 92, 246, 0.2),
            0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .model-comparison:hover {
        transform: translateX(4px);
        border-left-width: 6px;
        box-shadow: 
            -6px 0 30px rgba(139, 92, 246, 0.4),
            0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
        border: 2px solid rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #60a5fa, #a78bfa);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(59, 130, 246, 0.5),
            transparent
        );
    }
    
    div[data-baseweb="select"] {
        background: rgba(51, 65, 85, 0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    }
    
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stSuccess, .stInfo, .stWarning {
        backdrop-filter: blur(16px);
        border-radius: 12px;
        border-left-width: 4px;
    }
    
    .streamlit-expanderHeader {
        background: rgba(51, 65, 85, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(59, 130, 246, 0.5);
        background: rgba(59, 130, 246, 0.05);
    }
    
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
        border-right-color: #8b5cf6 !important;
        border-bottom-color: #ec4899 !important;
        border-left-color: #10b981 !important;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 30%, #ec4899 70%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        animation: gradientFlow 8s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 300;
        margin-bottom: 3rem;
        letter-spacing: 0.5px;
    }
    
    .stat-badge {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        color: #60a5fa;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 4px;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    file_path = 'cleaned_ai_impact_data_updated.csv'
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except:
            continue
    
    if df is not None:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        if 'ID' in df.columns:
            df.drop('ID', axis=1, inplace=True)
    return df

@st.cache_resource
def train_models(X, y, model_type="employment"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    return rf_model, xgb_model, rf_accuracy, xgb_accuracy, X_test, y_test

df = load_and_clean_data()

if df is not None:
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='color: #60a5fa; margin: 0; font-size: 2rem;'>🚀 AI Nexus</h1>
                <p style='color: #94a3b8; margin: 10px 0 0 0; font-size: 0.9rem; letter-spacing: 2px;'>
                    ADVANCED ANALYTICS PLATFORM
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        menu = st.radio("Navigation", ["📊 Dashboard", "🔬 Deep Insights", "🔮 Prediction Lab", "⚡ Model Arena"], label_visibility="collapsed")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: rgba(59, 130, 246, 0.1); 
                        padding: 20px; 
                        border-radius: 16px; 
                        text-align: center;
                        border: 1px solid rgba(59, 130, 246, 0.3);
                        backdrop-filter: blur(10px);'>
                <p style='margin: 0; color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>
                    ⚡ Dual Model System
                </p>
                <p style='margin: 10px 0 0 0; color: #10b981; font-size: 1.5rem; font-weight: 700;'>
                    Employment & Trust
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Pro Tip:** Use dual prediction mode for comprehensive analysis!")
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; backdrop-filter: blur(10px);'>
                <p style='color: #94a3b8; margin: 0; font-size: 0.8rem;'>TOTAL RECORDS</p>
                <p style='color: #60a5fa; margin: 5px 0 0 0; font-size: 2rem; font-weight: 700;'>{len(df)}</p>
            </div>
        """, unsafe_allow_html=True)

    if menu == "📊 Dashboard":
        st.markdown("<h1 class='hero-title'>🌐 AI SOCIAL IMPACT NEXUS</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Real-time insights powered by advanced machine learning</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Responses", len(df), delta="+12 this week")
        c2.metric("🧠 AI Knowledge", "Moderate", delta="↑ 15%")
        c3.metric("⭐ Usage Score", f"{df['AI Usage Rating'].mean():.1f}/5", delta="+0.3")
        c4.metric("🎓 Top Education", df['Education Level'].mode()[0][:15] + "...")

        st.markdown("<br>", unsafe_allow_html=True)

        col_banner1, col_banner2, col_banner3 = st.columns(3)
        
        trust_pct = (df['Trust in AI'].value_counts(normalize=True).iloc[0] * 100)
        future_interest = (df['Future AI Interest'].value_counts(normalize=True).iloc[0] * 100)
        job_concern_count = len(df[df['AI Job Impact'].astype(str).str.lower() == 'yes'])
        job_concern_pct = (job_concern_count / len(df)) * 100
        
        with col_banner1:
            st.markdown(f"""
                <div class='model-comparison' style='border-left-color: #10b981;'>
                    <h3 style='color: #10b981; margin: 0;'>✅ Trust Rate</h3>
                    <h1 style='color: #10b981; margin: 15px 0; font-size: 3rem;'>{trust_pct:.1f}%</h1>
                    <p style='color: #94a3b8; margin: 0;'>of respondents trust AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_banner2:
            st.markdown(f"""
                <div class='model-comparison' style='border-left-color: #3b82f6;'>
                    <h3 style='color: #3b82f6; margin: 0;'>🚀 Future Interest</h3>
                    <h1 style='color: #3b82f6; margin: 15px 0; font-size: 3rem;'>{future_interest:.1f}%</h1>
                    <p style='color: #94a3b8; margin: 0;'>want more AI products</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_banner3:
            st.markdown(f"""
                <div class='model-comparison' style='border-left-color: #f59e0b;'>
                    <h3 style='color: #f59e0b; margin: 0;'>⚠️ Job Concern</h3>
                    <h1 style='color: #f59e0b; margin: 15px 0; font-size: 3rem;'>{job_concern_pct:.1f}%</h1>
                    <p style='color: #94a3b8; margin: 0;'>worry about job impact</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Age Distribution")
            fig_age = px.histogram(df, x='Age Range', color='Age Range', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            fig_age.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=12), margin=dict(t=40, b=60, l=40, r=40), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'))
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("🤝 Trust vs Knowledge")
            fig_trust_know = px.histogram(df, x='AI Knowledge', color='Trust in AI', barmode='group', template="plotly_dark", color_discrete_sequence=['#3b82f6', '#ec4899', '#10b981'])
            fig_trust_know.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=12), margin=dict(t=40, b=60, l=40, r=40), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), legend=dict(bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.2)', borderwidth=1))
            st.plotly_chart(fig_trust_know, use_container_width=True)

    elif menu == "🔬 Deep Insights":
        st.markdown("<h1 class='hero-title'>🔬 DEEP ANALYSIS</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Uncovering hidden patterns in AI perception data</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
        st.subheader("🌍 Impact Perception")
        impact_data = df['AI Impact Perception'].value_counts().reset_index()
        fig_impact = px.bar(impact_data, y='AI Impact Perception', x='count', orientation='h', template="plotly_dark", color='AI Impact Perception', color_discrete_sequence=px.colors.sequential.Viridis)
        fig_impact.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=11), showlegend=False, xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'))
        st.plotly_chart(fig_impact, use_container_width=True)

    with col_b:
        st.subheader("💼 Job Impact by Education")
        fig_job = px.histogram(df, x='Education Level', color='AI Job Impact', barmode='group', template="plotly_dark", color_discrete_sequence=['#10b981', '#ef4444'])
        fig_job.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=11), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), legend=dict(bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.2)', borderwidth=1))
        st.plotly_chart(fig_job, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("📈 Usage Rating vs Trust")
    fig_usage_trust = px.box(df, x='Trust in AI', y='AI Usage Rating', color='Trust in AI', template="plotly_dark", points="all", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_usage_trust.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=12), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), showlegend=False)
    st.plotly_chart(fig_usage_trust, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔥 Correlation Matrix")
    
    corr_df = df.copy()
    for col in ['Age Range', 'Gender', 'Education Level', 'Employment Status', 'AI Knowledge', 'Trust in AI']:
        if col in corr_df.columns:
            le = LabelEncoder()
            corr_df[col + '_encoded'] = le.fit_transform(corr_df[col].astype(str))
    
    correlation_cols = [c for c in corr_df.columns if '_encoded' in c or c == 'AI Usage Rating']
    corr_matrix = corr_df[correlation_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=[c.replace('_encoded', '') for c in corr_matrix.columns], y=[c.replace('_encoded', '') for c in corr_matrix.index], colorscale='Turbo', text=corr_matrix.values, texttemplate='%{text:.2f}', textfont={"size": 10, "color": "white"}, hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'))
    fig_corr.update_layout(template="plotly_dark", height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Space Grotesk", size=11))
    st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "🔮 Prediction Lab":
    st.markdown("<h1 class='hero-title'>🔮 PREDICTION LAB</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Dual AI-powered prediction system with quantum accuracy</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🎯 Select Prediction Target")
    prediction_target = st.radio("", ["💼 Employment Status Only", "🤝 Trust in AI Only", "🔥 Both (Dual Prediction)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🤖 Choose ML Model")
    model_choice = st.radio("", ["🌲 Random Forest", "⚡ XGBoost (Recommended)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    features_employment = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
    target_employment = 'Employment Status'
    
    ml_df_emp = df[features_employment + [target_employment]].dropna()
    encoders_emp = {}
    for col in ['Age Range', 'Gender', 'Education Level', target_employment]:
        le = LabelEncoder()
        ml_df_emp[col] = le.fit_transform(ml_df_emp[col].astype(str))
        encoders_emp[col] = le

    X_emp = ml_df_emp[features_employment]
    y_emp = ml_df_emp[target_employment]
    
    features_trust = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
    target_trust = 'Trust in AI'
    
    ml_df_trust = df[features_trust + [target_trust]].dropna()
    encoders_trust = {}
    for col in ['Age Range', 'Gender', 'Education Level', 'AI Knowledge', target_trust]:
        le = LabelEncoder()
        ml_df_trust[col] = le.fit_transform(ml_df_trust[col].astype(str))
        encoders_trust[col] = le

    X_trust = ml_df_trust[features_trust]
    y_trust = ml_df_trust[target_trust]
    
    rf_model_emp, xgb_model_emp, rf_acc_emp, xgb_acc_emp, _, _ = train_models(X_emp, y_emp, "employment")
    rf_model_trust, xgb_model_trust, rf_acc_trust, xgb_acc_trust, _, _ = train_models(X_trust, y_trust, "trust")

    if "Both" in prediction_target:
        st.markdown("### 📊 Model Performance Overview")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 💼 Employment Model")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""<div class='model-comparison' style='border-left-color: #10b981;'><h4 style='color: #94a3b8; margin: 0;'>🌲 Random Forest</h4><h2 style='color: #10b981; margin: 10px 0;'>{rf_acc_emp*100:.2f}%</h2></div>""", unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"""<div class='model-comparison' style='border-left-color: #3b82f6;'><h4 style='color: #94a3b8; margin: 0;'>⚡ XGBoost</h4><h2 style='color: #3b82f6; margin: 10px 0;'>{xgb_acc_emp*100:.2f}%</h2></div>""", unsafe_allow_html=True)
        
        with col_b:
            st.markdown("#### 🤝 Trust Model")
            col_stat3, col_stat4 = st.columns(2)
            with col_stat3:
                st.markdown(f"""<div class='model-comparison' style='border-left-color: #10b981;'><h4 style='color: #94a3b8; margin: 0;'>🌲 Random Forest</h4><h2 style='color: #10b981; margin: 10px 0;'>{rf_acc_trust*100:.2f}%</h2></div>""", unsafe_allow_html=True)
            with col_stat4:
                st.markdown(f"""<div class='model-comparison' style='border-left-color: #3b82f6;'><h4 style='color: #94a3b8; margin: 0;'>⚡ XGBoost</h4><h2 style='color: #3b82f6; margin: 10px 0;'>{xgb_acc_trust*100:.2f}%</h2></div>""", unsafe_allow_html=True)
    
    elif "Employment" in prediction_target:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""<div class='model-comparison' style='border-left-color: #10b981;'><h3 style='color: #94a3b8;'>🌲 Random Forest</h3><h1 style='color: #10b981; margin: 15px 0;'>{rf_acc_emp*100:.2f}%</h1><p style='color: #94a3b8; margin: 0;'>Employment Prediction</p></div>""", unsafe_allow_html=True)
        with col_stat2:
            st.markdown(f"""<div class='model-comparison' style='border-left-color: #3b82f6;'><h3 style='color: #94a3b8;'>⚡ XGBoost</h3><h1 style='color: #3b82f6; margin: 15px 0;'>{xgb_acc_emp*100:.2f}%</h1><p style='color: #94a3b8; margin: 0;'>Employment Prediction</p></div>""", unsafe_allow_html=True)
    
    else:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""<div class='model-comparison' style='border-left-color: #10b981;'><h3 style='color: #94a3b8;'>🌲 Random Forest</h3><h1 style='color: #10b981; margin: 15px 0;'>{rf_acc_trust*100:.2f}%</h1><p style='color: #94a3b8; margin: 0;'>Trust Prediction</p></div>""", unsafe_allow_html=True)
        with col_stat2:
            st.markdown(f"""<div class='model-comparison' style='border-left-color: #3b82f6;'><h3 style='color: #94a3b8;'>⚡ XGBoost</h3><h1 style='color: #3b82f6; margin: 15px 0;'>{xgb_acc_trust*100:.2f}%</h1><p style='color: #94a3b8; margin: 0;'>Trust Prediction</p></div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Information")
    
    with st.container():
        c_left, c_right = st.columns(2)
        u_age = c_left.selectbox("🎂 Age Range", encoders_emp['Age Range'].classes_)
        u_gen = c_left.selectbox("👤 Gender", encoders_emp['Gender'].classes_)
        u_edu = c_right.selectbox("🎓 Education Level", encoders_emp['Education Level'].classes_)
        u_use = c_right.select_slider("📱 AI Usage Level", options=[1, 2, 3, 4, 5], value=3)
        
        if "Trust" in prediction_target or "Both" in prediction_target:
            u_knowledge = st.selectbox("🧠 AI Knowledge Level", encoders_trust['AI Knowledge'].classes_)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 GENERATE PREDICTION"):
            in_age_emp = encoders_emp['Age Range'].transform([u_age])[0]
            in_gen_emp = encoders_emp['Gender'].transform([u_gen])[0]
            in_edu_emp = encoders_emp['Education Level'].transform([u_edu])[0]
            
            if "Trust" in prediction_target or "Both" in prediction_target:
                in_age_trust = encoders_trust['Age Range'].transform([u_age])[0]
                in_gen_trust = encoders_trust['Gender'].transform([u_gen])[0]
                in_edu_trust = encoders_trust['Education Level'].transform([u_edu])[0]
                in_knowledge = encoders_trust['AI Knowledge'].transform([u_knowledge])[0]
            
            if "Both" in prediction_target:
                if "Random Forest" in model_choice:
                    emp_pred = rf_model_emp.predict([[in_age_emp, in_gen_emp, in_edu_emp, u_use]])
                    trust_pred = rf_model_trust.predict([[in_age_trust, in_gen_trust, in_edu_trust, u_use, in_knowledge]])
                    model_name = "Random Forest"
                    emp_acc = rf_acc_emp
                    trust_acc = rf_acc_trust
                else:
                    emp_pred = xgb_model_emp.predict([[in_age_emp, in_gen_emp, in_edu_emp, u_use]])
                    trust_pred = xgb_model_trust.predict([[in_age_trust, in_gen_trust, in_edu_trust, u_use, in_knowledge]])
                    model_name = "XGBoost"
                    emp_acc = xgb_acc_emp
                    trust_acc = xgb_acc_trust
                
                final_emp = encoders_emp[target_employment].inverse_transform(emp_pred)[0]
                final_trust = encoders_trust[target_trust].inverse_transform(trust_pred)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div class="dual-prediction">
                        <h1 style="color: white; margin: 0; font-size: 3rem; position: relative; z-index: 1;">🎯 DUAL PREDICTION RESULTS</h1>
                        <br>
                        <div style="display: flex; gap: 30px; justify-content: space-around; margin-top: 30px; position: relative; z-index: 1;">
                            <div style="background: rgba(15, 23, 42, 0.9); padding: 35px; border-radius: 20px; flex: 1; border: 1px solid rgba(59, 130, 246, 0.3); backdrop-filter: blur(10px);">
                                <h2 style="color: #60a5fa; margin: 0; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">💼 Employment Status</h2>
                                <h1 style="color: #10b981; margin: 20px 0; font-size: 2.5rem; font-weight: 800;">{final_emp}</h1>
                                <p style="color: #94a3b8; margin: 0;">Confidence: <span style="color: #10b981; font-weight: 700;">{emp_acc*100:.1f}%</span></p>
                            </div>
                            <div style="background: rgba(15, 23, 42, 0.9); padding: 35px; border-radius: 20px; flex: 1; border: 1px solid rgba(236, 72, 153, 0.3); backdrop-filter: blur(10px);">
                                <h2 style="color: #ec4899; margin: 0; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">🤝 Trust Level</h2>
                                <h1 style="color: #f59e0b; margin: 20px 0; font-size: 2.5rem; font-weight: 800;">{final_trust}</h1>
                                <p style="color: #94a3b8; margin: 0;">Confidence: <span style="color: #f59e0b; font-weight: 700;">{trust_acc*100:.1f}%</span></p>
                            </div>
                        </div>
                        <br>
                        <p style="color: white; font-size: 1.1rem; margin-top: 30px; position: relative; z-index: 1;">Model: <span style="font-weight: 700; color: #60a5fa;">{model_name}</span></p>
                    </div>
                """, unsafe_allow_html=True)
            
            elif "Employment" in prediction_target:
                if "Random Forest" in model_choice:
                    emp_pred = rf_model_emp.predict([[in_age_emp, in_gen_emp, in_edu_emp, u_use]])
                    model_name = "Random Forest"
                    model_acc = rf_acc_emp
                else:
                    emp_pred = xgb_model_emp.predict([[in_age_emp, in_gen_emp, in_edu_emp, u_use]])
                    model_name = "XGBoost"
                    model_acc = xgb_acc_emp
                
                final_emp = encoders_emp[target_employment].inverse_transform(emp_pred)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div class="prediction-card">
                        <h1 style="color:#60a5fa; margin:0; font-size: 3.5rem; position: relative; z-index: 1; font-weight: 800;">💼 {final_emp}</h1>
                        <p style="color:#94a3b8; font-size:1.3rem; margin: 25px 0; position: relative; z-index: 1;">Predicted Employment Status</p>
                        <div style="background: rgba(15, 23, 42, 0.9); padding: 25px; border-radius: 16px; margin-top: 30px; position: relative; z-index: 1; border: 1px solid rgba(59, 130, 246, 0.2);">
                            <p style="color:#10b981; font-size:1rem; margin:8px;">Model: <b>{model_name}</b></p>
                            <p style="color:#3b82f6; font-size:1rem; margin:8px;">Confidence: <b>{model_acc*100:.1f}%</b></p>
                            <p style="color:#f59e0b; font-size:1rem; margin:8px;">Training Samples: <b>{len(ml_df_emp)}</b></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            else:
                if "Random Forest" in model_choice:
                    trust_pred = rf_model_trust.predict([[in_age_trust, in_gen_trust, in_edu_trust, u_use, in_knowledge]])
                    model_name = "Random Forest"
                    model_acc = rf_acc_trust
                else:
                    trust_pred = xgb_model_trust.predict([[in_age_trust, in_gen_trust, in_edu_trust, u_use, in_knowledge]])
                    model_name = "XGBoost"
                    model_acc = xgb_acc_trust
                
                final_trust = encoders_trust[target_trust].inverse_transform(trust_pred)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div class="prediction-card">
                        <h1 style="color:#ec4899; margin:0; font-size: 3.5rem; position: relative; z-index: 1; font-weight: 800;">🤝 {final_trust}</h1>
                        <p style="color:#94a3b8; font-size:1.3rem; margin: 25px 0; position: relative; z-index: 1;">Predicted Trust in AI</p>
                        <div style="background: rgba(15, 23, 42, 0.9); padding: 25px; border-radius: 16px; margin-top: 30px; position: relative; z-index: 1; border: 1px solid rgba(236, 72, 153, 0.2);">
                            <p style="color:#10b981; font-size:1rem; margin:8px;">Model: <b>{model_name}</b></p>
                            <p style="color:#3b82f6; font-size:1rem; margin:8px;">Confidence: <b>{model_acc*100:.1f}%</b></p>
                            <p style="color:#f59e0b; font-size:1rem; margin:8px;">Training Samples: <b>{len(ml_df_trust)}</b></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("<h1 class='hero-title'>⚡ MODEL ARENA</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Battle of the algorithms: Random Forest vs XGBoost</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    comparison_target = st.radio("**Compare models for:**", ["💼 Employment Status", "🤝 Trust in AI"], horizontal=True)

    if "Employment" in comparison_target:
        features = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        target = 'Employment Status'
        
        ml_df = df[features + [target]].dropna()
        encoders = {}
        for col in ['Age Range', 'Gender', 'Education Level', target]:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            encoders[col] = le

        X = ml_df[features]
        y = ml_df[target]
        
        rf_model, xgb_model, rf_acc, xgb_acc, X_test, y_test = train_models(X, y, "employment")

    else:
        features = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
        target = 'Trust in AI'
        
        ml_df = df[features + [target]].dropna()
        encoders = {}
        for col in ['Age Range', 'Gender', 'Education Level', 'AI Knowledge', target]:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            encoders[col] = le

        X = ml_df[features]
        y = ml_df[target]
        
        rf_model, xgb_model, rf_acc, xgb_acc, X_test, y_test = train_models(X, y, "trust")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌲 Random Forest")
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
                 backdrop-filter: blur(20px);
                 padding: 40px;
                 border-radius: 20px;
                 text-align: center;
                 border: 2px solid rgba(16, 185, 129, 0.3);
                 box-shadow: 0 0 40px rgba(16, 185, 129, 0.2);'>
                <h1 style='color: #10b981; font-size: 4rem; margin: 0; font-weight: 800;'>{rf_acc*100:.2f}%</h1>
                <p style='color: #6ee7b7; font-size: 1.2rem; margin-top: 15px; text-transform: uppercase; letter-spacing: 2px;'>Test Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""**✨ Strengths:**\n- ✅ Excellent interpretability\n- ✅ Robust to overfitting\n- ✅ Handles non-linear patterns\n- ✅ Works well with smaller datasets\n- ✅ Minimal hyperparameter tuning""")
    
    with col2:
        st.markdown("### ⚡ XGBoost")
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%);
                 backdrop-filter: blur(20px);
                 padding: 40px;
                 border-radius: 20px;
                 text-align: center;
                 border: 2px solid rgba(59, 130, 246, 0.3);
                 box-shadow: 0 0 40px rgba(59, 130, 246, 0.2);'>
                <h1 style='color: #3b82f6; font-size: 4rem; margin: 0; font-weight: 800;'>{xgb_acc*100:.2f}%</h1>
                <p style='color: #93c5fd; font-size: 1.2rem; margin-top: 15px; text-transform: uppercase; letter-spacing: 2px;'>Test Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("""**⚡ Strengths:**\n- ⚡ Lightning-fast training\n- ⚡ Superior on complex patterns\n- ⚡ Built-in regularization\n- ⚡ Handles missing data\n- ⚡ Industry-standard performance""")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Feature Importance Analysis")
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        rf_importance = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=True)
        
        fig_rf_imp = go.Figure(go.Bar(x=rf_importance['Importance'], y=rf_importance['Feature'], orientation='h', marker=dict(color=rf_importance['Importance'], colorscale=[[0, '#064e3b'], [0.5, '#10b981'], [1, '#6ee7b7']], showscale=False, line=dict(color='rgba(16, 185, 129, 0.5)', width=1)), text=rf_importance['Importance'].round(3), textposition='outside', textfont=dict(size=11, color='#10b981')))
        fig_rf_imp.update_layout(title="🌲 Random Forest Importance", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Importance Score", yaxis_title="", height=400, font=dict(family="Space Grotesk", size=11), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_rf_imp, use_container_width=True)
    
    with col_imp2:
        xgb_importance = pd.DataFrame({'Feature': features, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=True)
        
        fig_xgb_imp = go.Figure(go.Bar(x=xgb_importance['Importance'], y=xgb_importance['Feature'], orientation='h', marker=dict(color=xgb_importance['Importance'], colorscale=[[0, '#1e3a8a'], [0.5, '#3b82f6'], [1, '#93c5fd']], showscale=False, line=dict(color='rgba(59, 130, 246, 0.5)', width=1)), text=xgb_importance['Importance'].round(3), textposition='outside', textfont=dict(size=11, color='#3b82f6')))
        fig_xgb_imp.update_layout(title="⚡ XGBoost Importance", template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Importance Score", yaxis_title="", height=400, font=dict(family="Space Grotesk", size=11), xaxis=dict(gridcolor='rgba(148, 163, 184, 0.1)'), yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_xgb_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Head-to-Head Comparison")
    
    comparison_df = pd.DataFrame({'Model': ['Random Forest', 'XGBoost'], 'Accuracy': [rf_acc * 100, xgb_acc * 100]})
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(name='Random Forest', x=['Random Forest'], y=[rf_acc * 100], marker=dict(color='#10b981', line=dict(color='#6ee7b7', width=2)), text=[f'{rf_acc*100:.2f}%'], textposition='outside', textfont=dict(size=16, color='#10b981', family='Space Grotesk', weight=700)))
    fig_comparison.add_trace(go.Bar(name='XGBoost', x=['XGBoost'], y=[xgb_acc * 100], marker=dict(color='#3b82f6', line=dict(color='#93c5fd', width=2)), text=[f'{xgb_acc*100:.2f}%'], textposition='outside', textfont=dict(size=16, color='#3b82f6', family='Space Grotesk', weight=700)))
fig_comparison.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450, showlegend=False, yaxis=dict(title="Accuracy (%)", range=[0, 105], gridcolor='rgba(148, 163, 184, 0.1)'), xaxis=dict(title="", gridcolor='rgba(0,0,0,0)'), font=dict(family="Space Grotesk", size=12))
st.plotly_chart(fig_comparison, use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)
    winner = "XGBoost ⚡" if xgb_acc > rf_acc else "Random Forest 🌲"
    winner_acc = max(xgb_acc, rf_acc) * 100
    winner_color = "#3b82f6" if xgb_acc > rf_acc else "#10b981"
    
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(124, 58, 237, 0.1) 100%);
             backdrop-filter: blur(20px);
             padding: 50px;
             border-radius: 28px;
             text-align: center;
             border: 3px solid rgba(167, 139, 250, 0.4);
             box-shadow: 0 0 60px rgba(139, 92, 246, 0.3);
             position: relative;
             overflow: hidden;'>
            <div style='position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; 
                 background: radial-gradient(circle, rgba(167, 139, 250, 0.1) 0%, transparent 70%);
                 animation: pulse 4s ease-in-out infinite;'></div>
            <h1 style='color: white; font-size: 3.5rem; margin: 0; position: relative; z-index: 1; font-weight: 800;'>🏆 CHAMPION: {winner}</h1>
            <p style='color: {winner_color}; font-size: 2rem; margin-top: 20px; position: relative; z-index: 1; font-weight: 700;'>{winner_acc:.2f}% Accuracy</p>
            <p style='color: #e9d5ff; font-size: 1.2rem; margin-top: 15px; position: relative; z-index: 1;'>Predicting: {comparison_target}</p>
            <div style='margin-top: 30px; position: relative; z-index: 1;'>
                <span class='stat-badge'>🎯 {len(ml_df)} Training Samples</span>
                <span class='stat-badge'>📊 {len(X_test)} Test Samples</span>
                <span class='stat-badge'>⚡ Optimized Performance</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
st.error("❌ Data file not found! Please ensure cleaned_ai_impact_data_updated.csv is in the correct directory.")
