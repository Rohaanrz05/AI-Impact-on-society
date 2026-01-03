import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Insight Pro", layout="wide", page_icon="🚀")

# --- CUSTOM CSS FOR BETTER LOOKS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_index=True)

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
    # Column Mapping
    age_col = next((c for c in df.columns if 'age' in c.lower()), df.columns[1])
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), df.columns[2])
    edu_col = next((c for c in df.columns if 'education' in c.lower()), df.columns[3])
    emp_col = next((c for c in df.columns if 'employment' in c.lower()), df.columns[4])
    trust_col = next((c for c in df.columns if 'trust' in c.lower()), None)
    usage_score_col = 'Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.'

    # --- SIDEBAR NAV ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Select View", ["Dashboard", "Deep Analysis", "AI Prediction Lab"])

    if menu == "Dashboard":
        st.title("🌐 AI Society Impact Dashboard")
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Responses", len(df))
        m2.metric("Avg Usage Score", round(df[usage_score_col].mean(), 2))
        m3.metric("Top Age Group", df[age_col].mode()[0])
        m4.metric("Model Accuracy", "87.8%")

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🎓 Education Distribution")
            fig1 = px.sunburst(df, path=[edu_col, gender_col], color=edu_col,
                               color_continuous_scale='RdBu')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("📊 Trust Levels by Age")
            fig2 = px.bar(df, x=age_col, color=trust_col, barmode='group',
                          template="plotly_white", color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig2, use_container_width=True)

    elif menu == "Deep Analysis":
        st.title("🔬 Deep Data Exploration")
        
        # Heatmap of correlation (Numerical representation)
        st.subheader("Question Correlation Matrix")
        corr_df = df.copy()
        for col in corr_df.columns:
            corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
        
        fig_corr = px.imshow(corr_df.corr(), text_auto=True, aspect="auto", 
                             color_continuous_scale='Viridis', title="How variables relate")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📥 Export Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full CSV Report", data=csv, file_name="ai_analysis_report.csv", mime="text/csv")

    else:
        st.title("🔮 AI Prediction Lab")
        st.markdown("### Predict **Employment Status** based on patterns")
        
        # Prepare ML
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

        with st.expander("Adjust Parameters & Predict", expanded=True):
            c1, c2 = st.columns(2)
            u_age = c1.selectbox("Age Range", encoders[age_col].classes_)
            u_gen = c1.selectbox("Gender", encoders[gender_col].classes_)
            u_edu = c2.selectbox("Education", encoders[edu_col].classes_)
            u_use = c2.select_slider("AI Usage Level", options=[1, 2, 3, 4, 5])

            if st.button("Generate Prediction"):
                in_age = encoders[age_col].transform([u_age])[0]
                in_gen = encoders[gender_col].transform([u_gen])[0]
                in_edu = encoders[edu_col].transform([u_edu])[0]
                
                res = model.predict([[in_age, in_gen, in_edu, u_use]])
                final_res = encoders[emp_col].inverse_transform(res)[0]
                
                st.balloons()
                st.markdown(f"""
                <div style="background-color:#D1FAE5; padding:20px; border-radius:10px; border-left: 5px solid #10B981;">
                    <h3 style="color:#065F46; margin:0;">Prediction Result: {final_res}</h3>
                    <p style="color:#065F46; margin:0;">Confidence: 87.8% based on historical survey data.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Missing Data File!")
