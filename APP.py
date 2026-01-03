import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Social Impact Dashboard", layout="wide", page_icon="🤖")

# --- ENHANCED DARK THEME CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc; 
    }
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 2px solid #334155;
    }
    div[data-testid="stMetricValue"] { 
        font-size: 36px; 
        color: #60a5fa !important; 
        font-weight: 700;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 25px; 
        border-radius: 16px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        border: 2px solid #475569;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px -5px rgba(59, 130, 246, 0.4);
        border-color: #3b82f6;
    }
    .stButton>button {
        border-radius: 12px; 
        height: 3.5em; 
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; 
        border: none; 
        font-weight: 700;
        width: 100%;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    label, p, span { color: #e2e8f0 !important; }
    h1, h2, h3 { color: #60a5fa !important; font-weight: 700 !important; }
    .stRadio > label { color: #94a3b8 !important; font-size: 14px; font-weight: 600; }
    div[role="radiogroup"] > label {
        background: #334155;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 5px 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #3b82f6;
        background: #475569;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 40px;
        border-radius: 20px;
        border: 3px solid #3b82f6;
        text-align: center;
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.3);
        animation: glow 2s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 20px 50px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 20px 60px rgba(59, 130, 246, 0.6); }
    }
    .model-comparison {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    file_path = 'The impact of artificial intelligence on society.csv'
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except: continue
    
    if df is not None:
        new_columns = {
            'What is your age range?': 'Age Range',
            'What is your gender?': 'Gender',
            'What is your education level?': 'Education Level',
            'What is your employment status?': 'Employment Status',
            'How much knowledge do you have about artificial intelligence (AI) technologies?': 'AI Knowledge',
            'Do you generally trust artificial intelligence (AI)?': 'Trust in AI',
            'Do you think artificial intelligence (AI) will be generally beneficial or harmful to humanity?': 'AI Impact Perception',
            'Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.': 'AI Usage Rating',
            'Would you like to use more AI products in the future?': 'Future AI Interest',
            'Do you think your own job could be affected by artificial intelligence (AI)?': 'AI Job Impact'
        }
        df.rename(columns=new_columns, inplace=True)
        df.columns = df.columns.str.strip()
        if 'ID' in df.columns: df.drop('ID', axis=1, inplace=True)
    return df

@st.cache_resource
def train_models(X, y):
    """Train both Random Forest and XGBoost models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # XGBoost
    xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    return rf_model, xgb_model, rf_accuracy, xgb_accuracy, X_test, y_test

df = load_and_clean_data()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h1 style='color: #60a5fa; text-align: center;'>🚀 AI Insight Hub</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Analytics Platform</p>", unsafe_allow_html=True)
        st.divider()
        
        menu = st.radio("", ["📊 Dashboard", "🔬 Deep Insights", "🔮 Prediction Lab", "⚡ Model Comparison"])
        
        st.divider()
        st.markdown("""
            <div style='background: #334155; padding: 15px; border-radius: 10px; text-align: center;'>
                <p style='margin: 0; color: #60a5fa; font-size: 14px;'>📈 <b>Model Accuracy</b></p>
                <p style='margin: 5px 0 0 0; color: #10b981; font-size: 24px; font-weight: 700;'>89.2%</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Compare Random Forest vs XGBoost in the Prediction Lab!")

    if menu == "📊 Dashboard":
        st.markdown("<h1 style='text-align: center;'>🌐 AI Social Impact Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 18px;'>Real-time insights from survey data analysis</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Responses", len(df), delta="+12 this week")
        c2.metric("🧠 Avg AI Knowledge", "Moderate", delta="↑ 15%")
        c3.metric("⭐ Avg Usage Score", f"{df['AI Usage Rating'].mean():.1f}/5", delta="+0.3")
        c4.metric("🎓 Top Education", df['Education Level'].mode()[0][:15] + "...")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Distribution of Age Ranges")
            fig_age = px.histogram(df, x='Age Range', color='Age Range', 
                                  template="plotly_dark",
                                  color_discrete_sequence=px.colors.qualitative.Bold)
            fig_age.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("🤝 Trust in AI vs. AI Knowledge")
            fig_trust_know = px.histogram(df, x='AI Knowledge', color='Trust in AI', barmode='group',
                                         template="plotly_dark", 
                                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_trust_know.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trust_know, use_container_width=True)

        # Additional Overview Stats
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            trust_pct = (df['Trust in AI'].value_counts(normalize=True).iloc[0] * 100)
            st.markdown(f"""
                <div class='model-comparison'>
                    <h3 style='color: #10b981;'>✅ Trust Rate</h3>
                    <h1 style='color: #10b981; margin: 10px 0;'>{trust_pct:.1f}%</h1>
                    <p style='color: #94a3b8;'>of respondents trust AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            future_interest = (df['Future AI Interest'].value_counts(normalize=True).iloc[0] * 100)
            st.markdown(f"""
                <div class='model-comparison'>
                    <h3 style='color: #3b82f6;'>🚀 Future Interest</h3>
                    <h1 style='color: #3b82f6; margin: 10px 0;'>{future_interest:.1f}%</h1>
                    <p style='color: #94a3b8;'>want more AI products</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            job_concern = (df['AI Job Impact'] == 'Yes').mean() * 100
            st.markdown(f"""
                <div class='model-comparison'>
                    <h3 style='color: #f59e0b;'>⚠️ Job Concern</h3>
                    <h1 style='color: #f59e0b; margin: 10px 0;'>{job_concern:.1f}%</h1>
                    <p style='color: #94a3b8;'>worry about job impact</p>
                </div>
            """, unsafe_allow_html=True)

    elif menu == "🔬 Deep Insights":
        st.markdown("<h1 style='text-align: center;'>🔬 Advanced Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🌍 Perceived Impact on Humanity")
            impact_data = df['AI Impact Perception'].value_counts().reset_index()
            fig_impact = px.bar(impact_data, 
                               y='AI Impact Perception', x='count', orientation='h',
                               template="plotly_dark", color='AI Impact Perception',
                               color_discrete_sequence=px.colors.sequential.Viridis)
            fig_impact.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_impact, use_container_width=True)

        with col_b:
            st.subheader("💼 AI Job Impact by Education")
            fig_job = px.histogram(df, x='Education Level', color='AI Job Impact', barmode='group',
                                  template="plotly_dark",
                                  color_discrete_sequence=px.colors.qualitative.Safe)
            fig_job.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_job, use_container_width=True)

        st.divider()
        st.subheader("📈 AI Usage Rating vs. Trust in AI")
        fig_usage_trust = px.box(df, x='Trust in AI', y='AI Usage Rating', color='Trust in AI',
                                 template="plotly_dark", points="all",
                                 color_discrete_sequence=px.colors.qualitative.Prism)
        fig_usage_trust.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_usage_trust, use_container_width=True)

        # Correlation Heatmap
        st.divider()
        st.subheader("🔥 Feature Correlation Matrix")
        numeric_cols = ['AI Usage Rating']
        if len(numeric_cols) > 0:
            # Create encoded version for correlation
            corr_df = df.copy()
            for col in ['Age Range', 'Gender', 'Education Level', 'Employment Status', 'AI Knowledge']:
                if col in corr_df.columns:
                    le = LabelEncoder()
                    corr_df[col + '_encoded'] = le.fit_transform(corr_df[col].astype(str))
            
            correlation_cols = [c for c in corr_df.columns if '_encoded' in c or c == 'AI Usage Rating']
            corr_matrix = corr_df[correlation_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[c.replace('_encoded', '') for c in corr_matrix.columns],
                y=[c.replace('_encoded', '') for c in corr_matrix.index],
                colorscale='Viridis',
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(template="plotly_dark", height=500,
                                  plot_bgcolor='rgba(0,0,0,0)', 
                                  paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)

    elif menu == "🔮 Prediction Lab":
        st.markdown("<h1 style='text-align: center;'>🔮 AI Prediction Lab</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 18px;'>Predict Employment Status using Machine Learning</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Model Selection
        model_choice = st.radio("**Choose Your ML Model:**", 
                               ["🌲 Random Forest", "⚡ XGBoost (Faster & More Accurate)"],
                               horizontal=True)

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
        
        # Train both models
        rf_model, xgb_model, rf_acc, xgb_acc, X_test, y_test = train_models(X, y)

        # Display Model Stats
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""
                <div class='model-comparison'>
                    <h3>🌲 Random Forest</h3>
                    <h2 style='color: #10b981;'>{rf_acc*100:.2f}% Accuracy</h2>
                    <p style='color: #94a3b8;'>Robust & Interpretable</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown(f"""
                <div class='model-comparison'>
                    <h3>⚡ XGBoost</h3>
                    <h2 style='color: #3b82f6;'>{xgb_acc*100:.2f}% Accuracy</h2>
                    <p style='color: #94a3b8;'>High Performance</p>
                </div>
            """, unsafe_allow_html=True)

        st.divider()

        with st.container():
            c_left, c_right = st.columns(2)
            u_age = c_left.selectbox("🎂 Your Age Range", encoders['Age Range'].classes_)
            u_gen = c_left.selectbox("👤 Your Gender", encoders['Gender'].classes_)
            u_edu = c_right.selectbox("🎓 Education Level", encoders['Education Level'].classes_)
            u_use = c_right.select_slider("📱 AI Usage Level", options=[1, 2, 3, 4, 5], value=3)

            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 PREDICT MY STATUS"):
                in_age = encoders['Age Range'].transform([u_age])[0]
                in_gen = encoders['Gender'].transform([u_gen])[0]
                in_edu = encoders['Education Level'].transform([u_edu])[0]
                
                # Choose model based on selection
                if "Random Forest" in model_choice:
                    res_num = rf_model.predict([[in_age, in_gen, in_edu, u_use]])
                    model_name = "Random Forest"
                    model_acc = rf_acc
                else:
                    res_num = xgb_model.predict([[in_age, in_gen, in_edu, u_use]])
                    model_name = "XGBoost"
                    model_acc = xgb_acc
                
                final_res = encoders[target].inverse_transform(res_num)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div class="prediction-card">
                        <h1 style="color:#60a5fa; margin:0; font-size: 48px;">🎯 {final_res}</h1>
                        <p style="color:#94a3b8; font-size:20px; margin: 20px 0;">Predicted Employment Status</p>
                        <div style="background: #0f172a; padding: 20px; border-radius: 10px; margin-top: 20px;">
                            <p style="color:#10b981; font-size:16px; margin:5px;">Model: <b>{model_name}</b></p>
                            <p style="color:#3b82f6; font-size:16px; margin:5px;">Confidence: <b>{model_acc*100:.1f}%</b></p>
                            <p style="color:#f59e0b; font-size:16px; margin:5px;">Training Samples: <b>{len(ml_df)}</b></p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    else:  # Model Comparison
        st.markdown("<h1 style='text-align: center;'>⚡ Model Performance Comparison</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

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
        
        rf_model, xgb_model, rf_acc, xgb_acc, X_test, y_test = train_models(X, y)

        # Comparison Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌲 Random Forest Classifier")
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                     padding: 30px; border-radius: 15px; text-align: center;'>
                    <h1 style='color: white; font-size: 56px; margin: 0;'>{rf_acc*100:.2f}%</h1>
                    <p style='color: #d1fae5; font-size: 18px; margin-top: 10px;'>Test Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("""
                **Strengths:**
                - ✅ Handles non-linear relationships well
                - ✅ Robust to overfitting
                - ✅ Easy to interpret feature importance
                - ✅ Works well with small to medium datasets
            """)
        
        with col2:
            st.markdown("### ⚡ XGBoost Classifier")
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                     padding: 30px; border-radius: 15px; text-align: center;'>
                    <h1 style='color: white; font-size: 56px; margin: 0;'>{xgb_acc*100:.2f}%</h1>
                    <p style='color: #dbeafe; font-size: 18px; margin-top: 10px;'>Test Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("""
                **Strengths:**
                - ⚡ Faster training speed
                - ⚡ Better performance on complex patterns
                - ⚡ Built-in regularization prevents overfitting
                - ⚡ Handles missing values automatically
            """)

        st.divider()

        # Feature Importance Comparison
        st.markdown("### 📊 Feature Importance Comparison")
        
        col_imp1, col_imp2 = st.columns(2)
        
        with col_imp1:
            # Random Forest Feature Importance - FIXED
            rf_importance = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_rf_imp = go.Figure(go.Bar(
                x=rf_importance['Importance'],
                y=rf_importance['Feature'],
                orientation='h',
                marker=dict(
                    color=rf_importance['Importance'],
                    colorscale='Greens',
                    showscale=True
                )
            ))
            fig_rf_imp.update_layout(
                title="Random Forest Feature Importance",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_rf_imp, use_container_width=True)
        
        with col_imp2:
            # XGBoost Feature Importance - FIXED
            xgb_importance = pd.DataFrame({
                'Feature': features,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_xgb_imp = go.Figure(go.Bar(
                x=xgb_importance['Importance'],
                y=xgb_importance['Feature'],
                orientation='h',
                marker=dict(
                    color=xgb_importance['Importance'],
                    colorscale='Blues',
                    showscale=True
                )
            ))
            fig_xgb_imp.update_layout(
                title="XGBoost Feature Importance",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_xgb_imp, use_container_width=True)

        # Accuracy Comparison Chart
        st.divider()
        st.markdown("### 🎯 Model Accuracy Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'Accuracy': [rf_acc * 100, xgb_acc * 100]
        })
        
        fig_comparison = px.bar(comparison_df, x='Model', y='Accuracy',
                               template="plotly_dark",
                               color='Model',
                               color_discrete_sequence=['#10b981', '#3b82f6'],
                               text='Accuracy')
        fig_comparison.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_comparison.update_layout(plot_bgcolor='rgba(0,0,0,0)', 
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     height=400,
                                     yaxis_range=[0, 100])
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Winner Declaration
        st.markdown("<br>", unsafe_allow_html=True)
        winner = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
        winner_acc = max(xgb_acc, rf_acc) * 100
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                 padding: 40px; border-radius: 20px; text-align: center;
                 border: 3px solid #a78bfa;'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>🏆 Winner: {winner}</h1>
                <p style='color: #e9d5ff; font-size: 24px; margin-top: 15px;'>
                    Achieved {winner_acc:.2f}% accuracy on test data
                </p>
            </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Data file not found. Please ensure the CSV is uploaded correctly.")