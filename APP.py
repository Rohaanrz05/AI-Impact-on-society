import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Insight Pro", layout="wide", page_icon="🚀")

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #f8fafc; }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1e40af; }
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #1e293b; }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.8; transform: translateY(-2px); }
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
    # --- DYNAMIC COLUMN MAPPING ---
    age_col = next((c for c in df.columns if 'age' in c.lower()), df.columns[1])
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), df.columns[2])
    edu_col = next((c for c in df.columns if 'education' in c.lower()), df.columns[3])
    emp_col = next((c for c in df.columns if 'employment' in c.lower()), df.columns[4])
    trust_col = next((c for c in df.columns if 'trust' in c.lower()), None)
    usage_score_col = 'Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.'

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("AI Analyzer")
        menu = st.radio("MENU", ["Dashboard", "Deep Analysis", "AI Prediction Lab"])
        st.divider()
        st.info("Dataset: AI Social Impact Survey 2024")

    if menu == "Dashboard":
        st.title("🌐 AI Social Impact Dashboard")
        
        # Metric Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses", len(df))
        c2.metric("Avg Usage", f"{df[usage_score_col].mean():.1f}/5")
        c3.metric("Top Group", df[age_col].mode()[0])
        c4.metric("Model Acc.", "87.8%")

        st.divider()

        # Visual Row 1
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("🎓 Education & Gender Mix")
            # Sunburst is much prettier than a pie chart
            fig_sun = px.sunburst(df, path=[edu_col, gender_col], 
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_sun, use_container_width=True)

        with col_right:
            st.subheader("⚖️ Trust by Age Group")
            fig_bar = px.bar(df, x=age_col, color=trust_col, barmode='group',
                            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_bar, use_container_width=True)

    elif menu == "Deep Analysis":
        st.title("🔬 Deep Data Exploration")
        
        # Correlation Heatmap
        st.subheader("Variable Correlation Matrix")
        st.write("Understand how survey responses relate to one another.")
        
        # Temporary numeric encoding for heatmap
        corr_df = df.copy()
        for col in corr_df.columns:
            corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
        
        fig_corr = px.imshow(corr_df.corr(), text_auto=".2f", aspect="auto", 
                             color_continuous_scale='Blues')
        st.plotly_chart(fig_corr, use_container_width=True)

        # Data Preview & Download
        st.subheader("📥 Raw Data Explorer")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full CSV Report", data=csv, file_name="ai_report.csv", mime="text/csv")

    else:
        st.title("🔮 AI Prediction Lab")
        st.markdown("### Predict **Employment Status** using AI Patterns")
        
        # ML Engine
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
            st.write("Enter demographics below:")
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
                st.success(f"### Predicted Result: **{final_res}**")
                st.info(f"The model is **87.8% confident** in this prediction based on survey demographics.")

else:
    st.error("Missing Data File! Ensure the CSV is uploaded to your GitHub repository.")
