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
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- 🎨 NEXT-GEN CSS THEME ---
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

    /* APP BACKGROUND & SCROLL */
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
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.8); }

    /* 🔮 INTERACTIVE GLASS CARDS (THE HOVER EFFECT) */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy effect */
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        transition: 0.5s;
    }

    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 
            0 20px 40px -10px rgba(99, 102, 241, 0.3),
            0 0 20px rgba(99, 102, 241, 0.2) inset; /* Inner glow */
    }
    
    .glass-card:hover::before {
        left: 100%;
    }

    /* METRIC HIGHLIGHTS */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* NEON BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.7);
    }

    /* HEADERS */
    h1, h2, h3 {
        color: white !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* DATAFRAME STYLING */
    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        overflow: hidden;
    }

    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def get_data():
    # Use the filename you uploaded
    file_path = 'cleaned_ai_impact_data_updated.csv'
    
    # Try different encodings
    for enc in ['utf-8', 'ISO-8859-1', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            
            # --- INTELLIGENT CLEANING ---
            df.columns = df.columns.str.strip()
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
            
            # Clean strings
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip()
                
            return df
        except:
            continue
    return None

# --- ML ENGINE ---
@st.cache_resource
def build_model(df, target_col, features):
    # Prepare Data
    ml_df = df[features + [target_col]].dropna()
    
    # Smart Encoding
    encoders = {}
    for col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le
        
    X = ml_df[features]
    y = ml_df[target_col]
    
    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    xgb = XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
    
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
    
    return rf, xgb, acc_rf, acc_xgb, encoders

df_raw = get_data()

# --- APP LAYOUT ---
if df_raw is not None:
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🧬 AI NEXUS</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # NAVIGATION
        page = st.radio("INTERFACE", [
            "🛸 Command Center", 
            "🔮 Prediction Lab", 
            "🧠 Deep Matrix", 
            "🛡️ Data Intelligence"
        ])
        
        st.markdown("---")
        
        # GLOBAL FILTER (New Feature!)
        st.markdown("### 🎛️ Global Filters")
        st.info("Adjusting these filters updates ALL charts.")
        
        filter_gender = st.multiselect("Gender", df_raw['Gender'].unique(), default=df_raw['Gender'].unique())
        filter_edu = st.multiselect("Education", df_raw['Education Level'].unique(), default=df_raw['Education Level'].unique())
        
        # APPLY FILTER
        df = df_raw[
            (df_raw['Gender'].isin(filter_gender)) & 
            (df_raw['Education Level'].isin(filter_edu))
        ]
        
        st.markdown(f"<div style='text-align:center; color:#64748b; margin-top:20px;'>Active Records: {len(df)}</div>", unsafe_allow_html=True)

    # --- PAGE 1: COMMAND CENTER (Dashboard) ---
    if page == "🛸 Command Center":
        st.markdown("<h1 class='gradient-text'>🛸 COMMAND CENTER</h1>", unsafe_allow_html=True)
        
        # 1. KPI ROW with Custom HTML Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        metrics = [
            ("Total Records", len(df), "Filtered View"),
            ("Avg Trust", f"{(df['Trust in AI'].str.contains('trust', case=False).mean()*100):.1f}%", "Trusting Users"),
            ("Usage Intensity", f"{df['AI Usage Rating'].astype(float).mean():.1f}/5", "Activity Level"),
            ("Top Concern", df['AI Job Impact'].mode()[0], "Job Security")
        ]
        
        for col, (label, val, sub) in zip([kpi1, kpi2, kpi3, kpi4], metrics):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 20px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # 2. MAIN CHARTS
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Demographics & Knowledge")
            fig = px.sunburst(df, path=['Gender', 'Age Range', 'AI Knowledge'], 
                              color='AI Knowledge', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=450, margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 Trust Distribution")
            fig2 = px.pie(df, names='Trust in AI', hole=0.6, color_discrete_sequence=px.colors.sequential.Bluyl)
            fig2.update_layout(height=450, margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=True, legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 2: PREDICTION LAB (Advanced) ---
    elif page == "🔮 Prediction Lab":
        st.markdown("<h1 class='gradient-text'>🔮 QUANTUM PREDICTOR</h1>", unsafe_allow_html=True)
        
        col_main, col_res = st.columns([1, 1.5])
        
        with col_main:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("👤 Profile Configuration")
            
            # Setup ML
            feats_emp = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating']
            rf_e, xgb_e, acc_e, _, enc_e = build_model(df_raw, 'Employment Status', feats_emp)
            
            feats_tru = ['Age Range', 'Gender', 'Education Level', 'AI Usage Rating', 'AI Knowledge']
            rf_t, xgb_t, acc_t, _, enc_t = build_model(df_raw, 'Trust in AI', feats_tru)
            
            # Inputs
            u_age = st.selectbox("Age", enc_e['Age Range'].classes_)
            u_gen = st.selectbox("Gender", enc_e['Gender'].classes_)
            u_edu = st.selectbox("Education", enc_e['Education Level'].classes_)
            u_use = st.slider("Usage (1-5)", 1, 5, 3)
            u_know = st.select_slider("Knowledge", options=enc_t['AI Knowledge'].classes_)
            
            predict = st.button("⚡ RUN SIMULATION")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            if predict:
                # Preprocessing
                in_e = [enc_e[c].transform([val])[0] for c, val in zip(feats_emp, [u_age, u_gen, u_edu, u_use])]
                in_t = [enc_t[c].transform([val])[0] for c, val in zip(feats_tru, [u_age, u_gen, u_edu, u_use, u_know])]
                
                # Prediction
                out_emp = enc_e['Employment Status'].inverse_transform(xgb_e.predict([in_e]))[0]
                out_tru = enc_t['Trust in AI'].inverse_transform(xgb_t.predict([in_t]))[0]
                
                # Result Display
                st.markdown(f"""
                <div style="display: flex; gap: 20px;">
                    <div class="glass-card" style="flex: 1; border-color: #10b981; text-align: center;">
                        <h3 style="color: #10b981 !important;">💼 Employment</h3>
                        <div class="metric-value" style="font-size: 2.5rem;">{out_emp}</div>
                        <div style="margin-top: 10px;">Model Confidence: {acc_e:.1%}</div>
                    </div>
                    <div class="glass-card" style="flex: 1; border-color: #ec4899; text-align: center;">
                        <h3 style="color: #ec4899 !important;">🤝 Trust Level</h3>
                        <div class="metric-value" style="font-size: 2.5rem;">{out_tru}</div>
                        <div style="margin-top: 10px;">Model Confidence: {acc_t:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature Importance Chart
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.caption("What influenced this decision?")
                imp = pd.DataFrame({'Feature': feats_tru, 'Importance': xgb_t.feature_importances_})
                fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                                 color='Importance', color_continuous_scale='Viridis')
                fig_imp.update_layout(height=200, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_imp, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👈 Configure the profile and hit 'RUN SIMULATION' to see AI predictions.")

    # --- PAGE 3: DEEP MATRIX (Advanced Charts) ---
    elif page == "🧠 Deep Matrix":
        st.markdown("<h1 class='gradient-text'>🧠 DEEP MATRIX INSIGHTS</h1>", unsafe_allow_html=True)
        
        # 3D Chart (New!)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧊 3D Analysis: Age vs Usage vs Trust")
        
        # Prepare data for 3D
        d3 = df.copy()
        le = LabelEncoder()
        d3['Trust_Code'] = le.fit_transform(d3['Trust in AI'])
        
        fig_3d = px.scatter_3d(
            d3, x='Age Range', y='AI Usage Rating', z='Trust_Code',
            color='Trust in AI', symbol='Gender',
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_3d.update_layout(
            height=600, 
            margin=dict(l=0, r=0, b=0, t=0),
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", color="white"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", color="white"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", color="white"),
            ),
            font_color="white"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Heatmap
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Variable Correlations")
        d_corr = df.apply(lambda x: pd.factorize(x)[0])
        corr = d_corr.corr()
        fig_h = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Magma'
        ))
        fig_h.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PAGE 4: DATA INTELLIGENCE (New Section) ---
    elif page == "🛡️ Data Intelligence":
        st.markdown("<h1 class='gradient-text'>🛡️ DATA INTELLIGENCE HUB</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧹 Dataset Health")
            missing = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()
            
            st.markdown(f"""
            <ul style="list-style: none; padding: 0;">
                <li style="margin: 10px 0;">🟢 <b>Missing Values:</b> {missing} (Clean)</li>
                <li style="margin: 10px 0;">🟡 <b>Duplicates Found:</b> {duplicates} (Expected in survey data)</li>
                <li style="margin: 10px 0;">🟣 <b>Total Columns:</b> {len(df.columns)}</li>
                <li style="margin: 10px 0;">🔵 <b>Total Rows:</b> {len(df)}</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### ⚖️ Class Balance: Trust")
            st.bar_chart(df['Trust in AI'].value_counts())
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Data Explorer
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Raw Data Explorer")
        with st.expander("📂 Click to view full dataset"):
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered CSV", csv, "ai_data_export.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("🚨 CRITICAL ERROR: Database Connection Failed. Please ensure 'cleaned_ai_impact_data_updated.csv' is in the root directory.")
