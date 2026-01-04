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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI Nexus | Ultra Dashboard", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING ---
st.markdown("""
    <style>
    /* FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.4);
    }

    /* GLOBAL APP STYLE */
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.15) 0px, transparent 50%), 
            radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.10) 0px, transparent 50%);
        font-family: 'Outfit', sans-serif;
    }

    /* 🔮 GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* METRIC HIGHLIGHTS */
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* CUSTOM SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* NEON BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }

    /* TEXT GRADIENTS */
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }

    /* CUSTOM TABS */
    div[data-testid="stRadio"] > div {
        background: transparent;
    }
    
    /* PLOTLY CHART CONTAINER */
    .js-plotly-plot {
        border-radius: 16px;
    }
    
    /* REMOVE PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ROBUST DATA LOADING ---
@st.cache_data
def get_dataset():
    file_path = 'cleaned_ai_impact_data_updated.csv'
    for enc in ['utf-8', 'ISO-8859-1', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df.columns = df.columns.str.strip()
            
            # Map friendly names
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
            
            # Clean all string columns
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
                
            return df
        except:
            continue
    return None

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def build_model(df, target_col, features):
    ml_df = df[features + [target_col]].dropna()
    encoders = {}
    
    # Robust Encoding
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

# Helper to prevent crashes on unseen inputs
def safe_transform(le, value):
    try:
        return le.transform([str(value)])[0]
    except:
        return 0 # Fallback

# --- 5. MAIN APPLICATION ---
df_raw = get_dataset()

if df_raw is not None:
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 0;'>🧬 AI NEXUS</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # NAV
        page = st.radio("MAIN INTERFACE", [
            "🛸 Command Center", 
            "🧠 Deep Matrix",
            "🔮 Prediction Lab", 
            "⚡ Model Arena"
        ])
        
        st.markdown("---")
        st.markdown("### 🎛️ Global Filters")
        
        # Dynamic Filters
        filter_gender = st.multiselect("Gender", df_raw['Gender'].unique(), default=df_raw['Gender'].unique())
        filter_edu = st.multiselect("Education", df_raw['Education Level'].unique(), default=df_raw['Education Level'].unique())
        
        # Apply Filters
        df = df_raw[
            (df_raw['Gender'].isin(filter_gender)) & 
            (df_raw['Education Level'].isin(filter_edu))
        ]
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 0.8rem;">ACTIVE RECORDS</div>
            <div style="color: #fff; font-size: 1.5rem; font-weight: 700;">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # TAB 1: COMMAND CENTER (Dashboard)
    # ==========================================
    if page == "🛸 Command Center":
        st.markdown("<h1 class='gradient-text'>COMMAND CENTER</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-top: -10px;'>Real-time AI perception analytics & key performance indicators.</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. METRICS ROW
        if 'AI Job Impact' in df.columns:
            job_concern_count = df[df['AI Job Impact'].str.lower() == 'yes'].shape[0]
        else:
            job_concern_count = 0

        k1, k2, k3, k4 = st.columns(4)
        
        def display_metric(col, title, value, sub, color="#6366f1"):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-bottom: 4px solid {color};">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        display_metric(k1, "Dataset Size", len(df), "Total Rows", "#6366f1")
        display_metric(k2, "Avg Trust", f"{(df['Trust in AI'].str.contains('trust', case=False).mean()*100):.0f}%", "Confidence Score", "#10b981")
        display_metric(k3, "Job Anxiety", f"{job_concern_count}", "Users Concerned", "#f59e0b")
        display_metric(k4, "Usage Score", f"{df['AI Usage Rating'].astype(float).mean():.1f}", "Avg Scale (1-5)", "#ec4899")

        # 2. MAIN CHARTS ROW
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Demographics Overview")
            # 2D Bar Chart
            fig_age = px.histogram(df, x='Age Range', color='Age Range', 
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_age.update_layout(height=350, margin=dict(t=20, l=0, r=0, b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False, font_color='white')
            st.plotly_chart(fig_age, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Breakdown")
            # 2D Donut
            fig_pie = px.pie(df, names='Trust in AI', hole=0.6, 
                             color_discrete_sequence=px.colors.sequential.Bluyl)
            fig_pie.update_layout(height=350, margin=dict(t=20, l=0, r=0, b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                  legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 2: DEEP MATRIX (Insights)
    # ==========================================
    elif page == "🧠 Deep Matrix":
        st.markdown("<h1 class='gradient-text'>DEEP MATRIX</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-top: -10px;'>Advanced correlation analysis and behavioral patterns.</p>", unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🫧 Usage Patterns by Trust")
            # 2D Bubble Chart
            fig_bubble = px.scatter(
                df, x='Age Range', y='AI Usage Rating',
                color='Trust in AI', size='AI Usage Rating',
                color_discrete_sequence=px.colors.qualitative.Bold,
                symbol='Gender'
            )
            fig_bubble.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🔥 Correlation Heatmap")
            # Correlation Matrix
            d_corr = df.apply(lambda x: pd.factorize(x)[0])
            corr = d_corr.corr()
            fig_hm = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale='Magma'
            ))
            fig_hm.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_hm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🌍 Impact Perception")
        fig_bar = px.bar(
            df['AI Impact Perception'].value_counts().reset_index(),
            x='count', y='AI Impact Perception', orientation='h',
            color='count', color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 3: PREDICTION LAB
    # ==========================================
    elif page == "🔮 Prediction Lab":
        st.markdown("<h1 class='gradient-text'>PREDICTION LAB</h1>", unsafe_allow_html=True)
        st.write("")

        col_input, col_res = st.columns([1, 1.5])
        
        with col_input:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👤 Subject Profile")
            
            # Init Models
            feats_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
            rf_e, xgb_e, acc_e, acc_xgb_e, enc_e = build_model(df_raw, 'Employment Status', feats_emp)
            
            feats_tru = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
            rf_t, xgb_t, acc_t, acc_xgb_t, enc_t = build_model(df_raw, 'Trust in AI', feats_tru)
            
            # Inputs
            model_choice = st.radio("Engine", ["⚡ XGBoost", "🌲 Random Forest"], horizontal=True)
            st.divider()
            
            u_age = st.selectbox("Age Range", enc_e['Age Range'].classes_)
            u_gen = st.selectbox("Gender", enc_e['Gender'].classes_)
            u_edu = st.selectbox("Education", enc_e['Education Level'].classes_)
            u_use = st.slider("Usage Intensity", 1, 5, 3)
            u_know = st.select_slider("Knowledge Level", options=enc_t['AI Knowledge'].classes_)
            
            run = st.button("🚀 Run Prediction Analysis")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            if run:
                # Prepare Safe Inputs
                in_e = [safe_transform(enc_e[c], val) for c, val in zip(feats_emp, [u_age, u_gen, u_edu, u_use])]
                in_t = [safe_transform(enc_t[c], val) for c, val in zip(feats_tru, [u_age, u_gen, u_edu, u_use, u_know])]
                
                if "XGBoost" in model_choice:
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

                # Result Cards
                st.markdown(f"""
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                    <div class="glass-card" style="flex: 1; text-align: center; border-top: 4px solid #10b981;">
                        <h4 style="color: #10b981; margin: 0;">💼 Employment Status</h4>
                        <div class="metric-value" style="font-size: 2.5rem; margin: 15px 0;">{pred_emp}</div>
                        <div style="color: #94a3b8;">Accuracy: {conf_emp:.1%}</div>
                    </div>
                    <div class="glass-card" style="flex: 1; text-align: center; border-top: 4px solid #ec4899;">
                        <h4 style="color: #ec4899; margin: 0;">🤝 Trust Level</h4>
                        <div class="metric-value" style="font-size: 2.5rem; margin: 15px 0;">{pred_tru}</div>
                        <div style="color: #94a3b8;">Accuracy: {conf_tru:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f" Analysis complete using **{m_name}** algorithm.")
            else:
                st.info("👈 Enter profile details and click 'Run Prediction Analysis'")

    # ==========================================
    # TAB 4: MODEL ARENA
    # ==========================================
    elif page == "⚡ Model Arena":
        st.markdown("<h1 class='gradient-text'>MODEL ARENA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; margin-top: -10px;'>Head-to-head comparison of ML algorithms.</p>", unsafe_allow_html=True)
        st.write("")

        target = st.selectbox("Select Target Variable", ["Employment Status", "Trust in AI"])
        
        feats = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge'] if target == 'Trust in AI' else ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
        rf, xgb, acc_rf, acc_xgb, _ = build_model(df_raw, target, feats)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #10b981;">
                <h3>🌲 Random Forest</h3>
                <div class="metric-value" style="color: #10b981 !important;">{acc_rf:.1%}</div>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #3b82f6;">
                <h3>⚡ XGBoost</h3>
                <div class="metric-value" style="color: #3b82f6 !important;">{acc_xgb:.1%}</div>
                <p>Accuracy Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Performance Visualization")
        
        # Comparison Bar Chart
        comp_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'], 
            'Accuracy': [acc_rf, acc_xgb]
        })
        fig_comp = px.bar(comp_df, x='Accuracy', y='Model', orientation='h',
                          color='Model', color_discrete_sequence=['#10b981', '#3b82f6'], text_auto='.1%')
        fig_comp.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("🚨 DATABASE CONNECTION ERROR: Please upload 'cleaned_ai_impact_data_updated.csv'")
