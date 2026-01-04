import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Impact | Ultra Dashboard",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- 🎨 ULTRA-MODERN CSS THEME ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* GLOBAL VARIABLES */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --bg-dark: #0f172a;
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-white: #f8fafc;
    }

    /* APP BACKGROUND & TYPOGRAPHY */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        background-attachment: fixed;
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, p, div, span {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-white);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--glass-border);
    }

    /* 🔮 GLASSMORPHISM CARDS */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* METRIC CARDS HTML */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .metric-value {
        font-size: 38px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* NEON BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); 
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }

    /* CUSTOM RADIO BUTTONS (SIDEBAR) */
    div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid transparent;
        border-radius: 10px;
        margin-bottom: 8px;
        padding: 10px 15px;
        transition: all 0.2s ease;
    }
    
    div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: var(--primary);
    }
    
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), transparent);
        border-left: 4px solid var(--accent);
    }

    /* HEADERS */
    .gradient-text {
        background: linear-gradient(to right, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* SLIDER CUSTOMIZATION */
    div[data-testid="stSlider"] > div {
        
    }
    
    /* PREDICTION RESULT BOX */
    .result-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: glow-pulse 3s infinite;
    }
    
    @keyframes glow-pulse {
        0% { box-shadow: 0 0 10px rgba(16, 185, 129, 0.1); }
        50% { box-shadow: 0 0 25px rgba(16, 185, 129, 0.4); }
        100% { box-shadow: 0 0 10px rgba(16, 185, 129, 0.1); }
    }

    </style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM CARD FUNCTION ---
def card_metric(label, value, subtext="", icon="📊"):
    st.markdown(f"""
        <div class="glass-card metric-container">
            <div style="font-size: 24px; margin-bottom: 5px;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 5px;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

# --- DATA LOADING (WITH YOUR FIXED LOGIC) ---
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
        # Mapping
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

# --- ML TRAINING CACHE ---
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    return rf, xgb, rf_acc, xgb_acc

# --- APP START ---
df = load_and_clean_data()

if df is not None:
    # --- SIDEBAR NAV ---
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🤖 AI NEXUS</h2>", unsafe_allow_html=True)
        st.write("Advanced Social Analytics Node")
        st.markdown("---")
        
        menu = st.radio("NAVIGATION", 
            ["Dashboard", "Deep Insights", "Prediction Lab", "Model Arena", "Data Explorer"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        st.success("● Models Online")
        st.info(f"● {len(df)} Records Loaded")

    # --- MAIN CONTENT ---
    
    # 1. DASHBOARD
    if menu == "Dashboard":
        st.markdown("<h1 class='gradient-text'>🌐 Global Overview</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8;'>Real-time analysis of Artificial Intelligence perception.</p>", unsafe_allow_html=True)
        
        # Ultra Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: card_metric("Total Responses", len(df), "Updated live", "📋")
        with c2: card_metric("Avg Usage", f"{df['AI Usage Rating'].mean():.1f}/5", "Out of 5.0", "⭐")
        with c3: card_metric("Trust Factor", f"{(df['Trust in AI'].value_counts(normalize=True).iloc[0]*100):.0f}%", "Trusting Population", "🛡️")
        with c4: card_metric("Knowledge Base", "Moderate", "Avg. User Level", "🧠")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Glass Containers for Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Demographics: Age")
            if 'Age Range' in df.columns:
                fig = px.pie(df, names='Age Range', hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            if 'Trust in AI' in df.columns:
                fig = px.bar(df['Trust in AI'].value_counts(), orientation='h', 
                             color_discrete_sequence=['#6366f1'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. DEEP INSIGHTS
    elif menu == "Deep Insights":
        st.markdown("<h1 class='gradient-text'>🔬 Deep Dive Analytics</h1>", unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔥 Multi-Variable Correlation Heatmap")
        
        # Prepare Heatmap
        corr_df = df.copy()
        for col in ['Age Range', 'Gender', 'Education Level', 'Employment Status', 'AI Knowledge', 'Trust in AI']:
            if col in corr_df.columns:
                corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
        
        numeric_cols = corr_df.select_dtypes(include=[int, float]).columns
        corr = corr_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Magma', text=corr.values, texttemplate='%{text:.2f}'
        ))
        fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎓 Education vs Fear of Job Loss")
            fig2 = px.histogram(df, x="Education Level", color="AI Job Impact", barmode="group",
                                color_discrete_sequence=px.colors.qualitative.Safe)
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # 3. PREDICTION LAB
    elif menu == "Prediction Lab":
        st.markdown("<h1 class='gradient-text'>🔮 Future Prediction Engine</h1>", unsafe_allow_html=True)
        
        # Setup Data
        features_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        ml_df_emp = df[features_emp + ['Employment Status']].dropna()
        
        features_trust = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
        ml_df_trust = df[features_trust + ['Trust in AI']].dropna()

        # Encoders
        enc_emp = {col: LabelEncoder().fit(ml_df_emp[col].astype(str)) for col in features_emp + ['Employment Status']}
        enc_trust = {col: LabelEncoder().fit(ml_df_trust[col].astype(str)) for col in features_trust + ['Trust in AI']}

        # Prepare X, y
        for col in features_emp: ml_df_emp[col] = enc_emp[col].transform(ml_df_emp[col].astype(str))
        ml_df_emp['Employment Status'] = enc_emp['Employment Status'].transform(ml_df_emp['Employment Status'].astype(str))
        
        for col in features_trust: ml_df_trust[col] = enc_trust[col].transform(ml_df_trust[col].astype(str))
        ml_df_trust['Trust in AI'] = enc_trust['Trust in AI'].transform(ml_df_trust['Trust in AI'].astype(str))

        # Train
        rf_e, xgb_e, _, _ = train_models(ml_df_emp[features_emp], ml_df_emp['Employment Status'])
        rf_t, xgb_t, _, _ = train_models(ml_df_trust[features_trust], ml_df_trust['Trust in AI'])

        # Input Section
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👤 Subject Profile")
        c1, c2, c3 = st.columns(3)
        age = c1.selectbox("Age Range", enc_emp['Age Range'].classes_)
        gender = c2.selectbox("Gender", enc_emp['Gender'].classes_)
        edu = c3.selectbox("Education", enc_emp['Education Level'].classes_)
        usage = st.slider("AI Usage Intensity", 1, 5, 3)
        knowledge = st.select_slider("AI Knowledge Level", options=enc_trust['AI Knowledge'].classes_)
        
        predict_btn = st.button("🚀 INITIALIZE PREDICTION SEQUENCE")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner('⚡ Calculating Probabilities...'):
                # Prepare Inputs
                input_emp = [enc_emp[c].transform([val])[0] for c, val in zip(features_emp, [age, gender, edu, usage])]
                input_trust = [enc_trust[c].transform([val])[0] for c, val in zip(features_trust, [age, gender, edu, usage, knowledge])]
                
                # Predict (Using XGBoost as default "Ultra" model)
                pred_emp = enc_emp['Employment Status'].inverse_transform(xgb_e.predict([input_emp]))[0]
                pred_trust = enc_trust['Trust in AI'].inverse_transform(xgb_t.predict([input_trust]))[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                
                with r1:
                    st.markdown(f"""
                    <div class="result-box">
                        <h3 style="color:#60a5fa; margin:0;">💼 Employment Status</h3>
                        <h1 style="font-size:42px; color:white; margin:10px 0;">{pred_emp}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with r2:
                    st.markdown(f"""
                    <div class="result-box" style="border-color: #f59e0b; box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);">
                        <h3 style="color:#f59e0b; margin:0;">🤝 Trust Level</h3>
                        <h1 style="font-size:42px; color:white; margin:10px 0;">{pred_trust}</h1>
                    </div>
                    """, unsafe_allow_html=True)

    # 4. MODEL ARENA
    elif menu == "Model Arena":
        st.markdown("<h1 class='gradient-text'>⚔️ Model Performance Arena</h1>", unsafe_allow_html=True)
        
        # Simulating data again for quick access
        features = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        ml_df = df[features + ['Employment Status']].dropna()
        le_dict = {c: LabelEncoder().fit(ml_df[c].astype(str)) for c in ml_df.columns}
        for c in ml_df.columns: ml_df[c] = le_dict[c].transform(ml_df[c].astype(str))
        
        rf, xgb, rf_acc, xgb_acc = train_models(ml_df[features], ml_df['Employment Status'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid #10b981;">
                <h2>🌲 Random Forest</h2>
                <h1 style="color: #10b981; font-size: 50px;">{rf_acc*100:.2f}%</h1>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid #3b82f6;">
                <h2>⚡ XGBoost</h2>
                <h1 style="color: #3b82f6; font-size: 50px;">{xgb_acc*100:.2f}%</h1>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Feature Importance
        st.markdown("### 🧠 Feature Importance Architecture")
        imp = pd.DataFrame({'Feature': features, 'Importance': xgb.feature_importances_}).sort_values('Importance', ascending=True)
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h', template="plotly_dark",
                     color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # 5. DATA EXPLORER (New Feature)
    elif menu == "Data Explorer":
        st.markdown("<h1 class='gradient-text'>💾 Raw Data Vault</h1>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        with st.expander("🔎 Filter & Search Options", expanded=True):
            cols = st.multiselect("Select Columns to View", df.columns.tolist(), default=df.columns[:5].tolist())
            search = st.text_input("Search Keyword (e.g., Student, Yes)")
        
        display_df = df[cols] if cols else df
        
        if search:
            mask = display_df.apply(lambda x: x.astype(str).str.contains(search, case=False)).any(axis=1)
            display_df = display_df[mask]
            
        st.dataframe(display_df, use_container_width=True, height=500)
        
        # Download Button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="ai_impact_data_export.csv",
            mime="text/csv",
        )
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("⚠️ DATA CONNECTION ERROR: Please ensure 'cleaned_ai_impact_data_updated.csv' is in the directory.")
