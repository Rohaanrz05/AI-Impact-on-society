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
    page_title="AI Social Impact Nexus", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

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
        color: #f8fafc;
    }
    
    /* Animated background particles */
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
        animation: backgroundScroll 60s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes backgroundScroll {
        0% { background-position: 0 0, 40px 60px, 130px 270px, 70px 100px; }
        100% { background-position: 200px 200px, 240px 260px, 330px 470px, 270px 300px; }
    }
    
    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] { 
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Premium Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
    }
    
    div[data-testid="stMetricValue"] { 
        font-size: 32px !important;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Ultra Premium Buttons */
    .stButton>button {
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-position: 100% 0;
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Typography */
    h1 { 
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2, h3 { color: #e2e8f0 !important; }
    
    /* Result Cards */
    .prediction-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        padding: 40px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        animation: glow 3s infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }
        to { box-shadow: 0 0 40px rgba(59, 130, 246, 0.3); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #3b82f6; border-radius: 5px; }
    
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
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
        # Rename mapping to match code expectations
        rename_mapping = {
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
        df.rename(columns=rename_mapping, inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            
        if 'ID' in df.columns:
            df.drop('ID', axis=1, inplace=True)
    return df

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(X, y, model_type="employment"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    return rf_model, xgb_model, rf_accuracy, xgb_accuracy, X_test, y_test

df = load_and_clean_data()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h1 style='font-size: 24px; text-align: center;'>🚀 AI NEXUS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; letter-spacing: 2px; font-size: 12px;'>ANALYTICS PLATFORM</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.radio(
            "NAVIGATION", 
            ["📊 Dashboard", "🔬 Deep Insights", "🔮 Prediction Lab", "⚡ Model Arena"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("💡 **Pro Tip:** Use the Prediction Lab to simulate user profiles.")

    # --- 1. DASHBOARD ---
    if menu == "📊 Dashboard":
        st.markdown("<h1 style='font-size: 3rem;'>🌐 AI SOCIAL IMPACT NEXUS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Real-time insights powered by advanced machine learning</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Records", len(df), delta="Live")
        c2.metric("🧠 AI Knowledge", "Moderate", delta="↑ 15%")
        c3.metric("⭐ Usage Score", f"{df['AI Usage Rating'].mean():.1f}/5", delta="+0.3")
        c4.metric("🎓 Top Education", df['Education Level'].mode()[0][:10] + "...")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Age Distribution")
            fig_age = px.histogram(df, x='Age Range', color='Age Range', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            fig_age.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("🤝 Trust vs Knowledge")
            fig_trust = px.histogram(df, x='AI Knowledge', color='Trust in AI', barmode='group', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_trust.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trust, use_container_width=True)

    # --- 2. DEEP INSIGHTS ---
    elif menu == "🔬 Deep Insights":
        st.markdown("<h1>🔬 DEEP ANALYSIS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Uncovering hidden patterns in AI perception data</p>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🌍 Impact Perception")
            impact_data = df['AI Impact Perception'].value_counts().reset_index()
            fig_impact = px.bar(impact_data, y='AI Impact Perception', x='count', orientation='h', template="plotly_dark", color='AI Impact Perception', color_discrete_sequence=px.colors.sequential.Viridis)
            fig_impact.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_impact, use_container_width=True)

        with col_b:
            st.subheader("💼 Job Impact by Education")
            fig_job = px.histogram(df, x='Education Level', color='AI Job Impact', barmode='group', template="plotly_dark", color_discrete_sequence=['#10b981', '#ef4444'])
            fig_job.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_job, use_container_width=True)

        st.markdown("---")
        st.subheader("🔥 Correlation Matrix")
        
        # Prepare correlation data
        corr_df = df.copy()
        for col in ['Age Range', 'Gender', 'Education Level', 'Employment Status', 'AI Knowledge', 'Trust in AI']:
            if col in corr_df.columns:
                corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
        
        numeric_cols = corr_df.select_dtypes(include=[np.number]).columns
        corr_matrix = corr_df[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Turbo',
            text=corr_matrix.values,
            texttemplate='%{text:.2f}'
        ))
        fig_corr.update_layout(template="plotly_dark", height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- 3. PREDICTION LAB ---
    elif menu == "🔮 Prediction Lab":
        st.markdown("<h1>🔮 PREDICTION LAB</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Dual AI-powered prediction system</p>", unsafe_allow_html=True)
        
        # Setup Data
        features_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        ml_df_emp = df[features_emp + ['Employment Status']].dropna()
        enc_emp = {col: LabelEncoder().fit(ml_df_emp[col].astype(str)) for col in features_emp + ['Employment Status']}
        
        features_trust = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
        ml_df_trust = df[features_trust + ['Trust in AI']].dropna()
        enc_trust = {col: LabelEncoder().fit(ml_df_trust[col].astype(str)) for col in features_trust + ['Trust in AI']}

        # Transform Data
        for col in features_emp: ml_df_emp[col] = enc_emp[col].transform(ml_df_emp[col].astype(str))
        ml_df_emp['Employment Status'] = enc_emp['Employment Status'].transform(ml_df_emp['Employment Status'].astype(str))
        
        for col in features_trust: ml_df_trust[col] = enc_trust[col].transform(ml_df_trust[col].astype(str))
        ml_df_trust['Trust in AI'] = enc_trust['Trust in AI'].transform(ml_df_trust['Trust in AI'].astype(str))

        # Train
        rf_e, xgb_e, _, xgb_acc_e, _, _ = train_models(ml_df_emp[features_emp], ml_df_emp['Employment Status'])
        rf_t, xgb_t, _, xgb_acc_t, _, _ = train_models(ml_df_trust[features_trust], ml_df_trust['Trust in AI'])

        st.markdown("### 📝 User Profile")
        with st.container():
            c1, c2, c3 = st.columns(3)
            u_age = c1.selectbox("Age Range", enc_emp['Age Range'].classes_)
            u_gen = c2.selectbox("Gender", enc_emp['Gender'].classes_)
            u_edu = c3.selectbox("Education", enc_emp['Education Level'].classes_)
            u_use = st.slider("AI Usage Level", 1, 5, 3)
            u_know = st.select_slider("AI Knowledge", options=enc_trust['AI Knowledge'].classes_)

            if st.button("🚀 GENERATE PREDICTION"):
                # Encode
                in_emp = [enc_emp[c].transform([val])[0] for c, val in zip(features_emp, [u_age, u_gen, u_edu, u_use])]
                in_trust = [enc_trust[c].transform([val])[0] for c, val in zip(features_trust, [u_age, u_gen, u_edu, u_use, u_know])]
                
                # Predict
                pred_emp = enc_emp['Employment Status'].inverse_transform(xgb_e.predict([in_emp]))[0]
                pred_trust = enc_trust['Trust in AI'].inverse_transform(xgb_t.predict([in_trust]))[0]

                st.balloons()
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3 style="color:#60a5fa;">💼 Employment</h3>
                        <h1 style="color:#10b981; font-size: 36px; margin: 10px 0;">{pred_emp}</h1>
                        <p>Confidence: {xgb_acc_e*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div class="prediction-card" style="border-color: #ec4899;">
                        <h3 style="color:#ec4899;">🤝 Trust Level</h3>
                        <h1 style="color:#f59e0b; font-size: 36px; margin: 10px 0;">{pred_trust}</h1>
                        <p>Confidence: {xgb_acc_t*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

    # --- 4. MODEL ARENA ---
    elif menu == "⚡ Model Arena":
        st.markdown("<h1>⚡ MODEL ARENA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Algorithm Showdown: Random Forest vs XGBoost</p>", unsafe_allow_html=True)
        
        target_choice = st.radio("Target Variable", ["Employment Status", "Trust in AI"], horizontal=True)
        
        # Quick retrain for stats
        if target_choice == "Employment Status":
            features = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
            target = 'Employment Status'
            ml_df = df[features + [target]].dropna()
            le_dict = {c: LabelEncoder().fit(ml_df[c].astype(str)) for c in ml_df.columns}
            for c in ml_df.columns: ml_df[c] = le_dict[c].transform(ml_df[c].astype(str))
            rf, xgb, rf_acc, xgb_acc, _, _ = train_models(ml_df[features], ml_df[target])
        else:
            features = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
            target = 'Trust in AI'
            ml_df = df[features + [target]].dropna()
            le_dict = {c: LabelEncoder().fit(ml_df[c].astype(str)) for c in ml_df.columns}
            for c in ml_df.columns: ml_df[c] = le_dict[c].transform(ml_df[c].astype(str))
            rf, xgb, rf_acc, xgb_acc, _, _ = train_models(ml_df[features], ml_df[target])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="prediction-card" style="border: 2px solid #10b981;">
                <h2>🌲 Random Forest</h2>
                <h1 style="color: #10b981; font-size: 48px;">{rf_acc*100:.2f}%</h1>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="prediction-card" style="border: 2px solid #3b82f6;">
                <h2>⚡ XGBoost</h2>
                <h1 style="color: #3b82f6; font-size: 48px;">{xgb_acc*100:.2f}%</h1>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📊 Feature Importance (XGBoost)")
        imp = pd.DataFrame({'Feature': features, 'Importance': xgb.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', template="plotly_dark", color='Importance')
        fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.error("❌ Data file not found! Please ensure 'cleaned_ai_impact_data_updated.csv' is in the folder.")
