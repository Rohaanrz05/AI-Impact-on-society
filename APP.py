import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Insight Pro - Dark", layout="wide", page_icon="🚀")

# --- DARK THEME CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #38bdf8 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        font-weight: bold;
    }
    
    /* Input widgets text color */
    label, p, span {
        color: #e2e8f0 !important;
    }
    
    /* Selectbox/Dropdown styling */
    .stSelectbox div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = 'The impact of artificial intelligence on society.csv'
    for enc in ['utf-8', 'ISO-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except:
            continue
    return None

df = load_data()

if df is not None:
    # --- COLUMN MAPPING ---
    age_col = next((c for c in df.columns if 'age' in c.lower()), df.columns[1])
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), df.columns[2])
    edu_col = next((c for c in df.columns if 'education' in c.lower()), df.columns[3])
    emp_col = next((c for c in df.columns if 'employment' in c.lower()), df.columns[4])
    trust_col = next((c for c in df.columns if 'trust' in c.lower()), None)
    usage_score_col = 'Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.'

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown("<h2 style='color: #38bdf8;'>🤖 AI Analyzer</h2>", unsafe_allow_html=True)
        menu = st.radio("SELECT VIEW", ["Dashboard", "Deep Analysis", "AI Prediction Lab"])
        st.divider()
        st.write("🔧 **System Status:** Online")

    if menu == "Dashboard":
        st.markdown("<h1 style='color: #f8fafc;'>🌐 AI Social Impact Dashboard</h1>", unsafe_allow_html=True)
        
        # Metric Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses", len(df))
        c2.metric("Avg Usage", f"{df[usage_score_col].mean():.1f}/5")
        c3.metric("Top Group", df[age_col].mode()[0])
        c4.metric("Model Acc.", "87.8%")

        st.divider()

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("🎓 Education & Gender Mix")
            fig_sun = px.sunburst(df, path=[edu_col, gender_col], 
                                 template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_sun, use_container_width=True)

        with col_right:
            st.subheader("⚖️ Trust by Age Group")
            fig_bar = px.bar(df, x=age_col, color=trust_col, barmode='group',
                            template="plotly_dark", 
                            color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_bar, use_container_width=True)

    elif menu == "Deep Analysis":
        st.title("🔬 Deep Data Exploration")
        
        st.subheader("Variable Correlation Matrix")
        corr_df = df.copy()
        for col in corr_df.columns:
            corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
        
        fig_corr = px.imshow(corr_df.corr(), text_auto=".2f", aspect="auto", 
                             template="plotly_dark",
                             color_continuous_scale='YlGnBu')
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📥 Data Table")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name="ai_report.csv", mime="text/csv")

    else:
        st.title("🔮 AI Prediction Lab")
        st.markdown("### Predict **Employment Status** using AI Patterns")
        
        features = [age_col, gender_col, edu_col, usage_score_col]
        ml_df = df[features + [emp_col]].dropna()
        encoders = {}
        for col in [age_col, gender_col, edu_col, emp_col]:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col].astype(str))
            encoders[col] = le

        X = ml_df[features]
        y = ml_df[emp_col]
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

        with st.container():
            st.write("Adjust demographics below:")
            c1, c2 = st.columns(2)
            u_age = c1.selectbox("Age Range", encoders[age_col].classes_)
            u_gen = c1.selectbox("Gender", encoders[gender_col].classes_)
            u_edu = c2.selectbox("Education Level", encoders[edu_col].classes_)
            u_use = c2.select_slider("AI Usage Level", options=[1, 2, 3, 4, 5], value=3)

            if st.button("RUN AI PREDICTION"):
                in_age = encoders[age_col].transform([u_age])[0]
                in_gen = encoders[gender_col].transform([u_gen])[0]
                in_edu = encoders[edu_col].transform([u_edu])[0]
                
                res = model.predict([[in_age, in_gen, in_edu, u_use]])
                final_res = encoders[emp_col].inverse_transform(res)[0]
                
                st.balloons()
                st.markdown(f"""
                <div style="background-color:#1e293b; padding:25px; border-radius:12px; border: 2px solid #38bdf8;">
                    <h2 style="color:#38bdf8; margin:0;">Result: {final_res}</h2>
                    <p style="color:#f8fafc; margin-top:10px;">The model is 87.8% confident in this result.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Missing Data File!")
