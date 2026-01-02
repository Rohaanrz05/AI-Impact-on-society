import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="AI & Society Dashboard", layout="wide")

# Load data
df = pd.read_csv('The impact of artificial intelligence on society.csv')

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Results")
gender_filter = st.sidebar.multiselect("Select Gender:", options=df['What is your gender?'].unique(), default=df['What is your gender?'].unique())
age_filter = st.sidebar.multiselect("Select Age Range:", options=df['What is your age range?'].unique(), default=df['What is your age range?'].unique())

# Filtered Data
df_selection = df.query("`What is your gender?` == @gender_filter & `What is your age range?` == @age_filter")

# --- MAIN PAGE ---
st.title("📊 The Impact of AI on Society")
st.markdown("Exploring public perception, trust, and the future of AI.")

tabs = st.tabs(["📈 Data Insights", "🔮 Predictor Tool"])

with tabs[0]:
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    avg_usage = df_selection['Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.'].mean()
    trust_pct = (df_selection['Do you generally trust artificial intelligence (AI)?'] == 'I trust it').mean() * 100

    col1.metric("Total Respondents", len(df_selection))
    col2.metric("Avg. AI Usage Score", f"{avg_usage:.2f}/5")
    col3.metric("Trust AI", f"{trust_pct:.1f}%")

    # Visualizations
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        fig_trust = px.histogram(df_selection, x="Do you generally trust artificial intelligence (AI)?", 
                                 title="General Trust in AI", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_trust, use_container_width=True)
        
    with c2:
        fig_edu = px.box(df_selection, x="What is your education level?", 
                         y="Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.",
                         title="AI Usage by Education Level")
        st.plotly_chart(fig_edu, use_container_width=True)

with tabs[1]:
    st.subheader("Predict Your AI Sentiment")
    st.info("Input your demographics below to see your predicted 'AI Usage Level' based on our ML model.")
    
    with st.form("predictor_form"):
        age = st.selectbox("Age Range", df['What is your age range?'].unique())
        edu = st.selectbox("Education Level", df['What is your education level?'].unique())
        job = st.selectbox("Employment Status", df['What is your employment status?'].unique())
        submit = st.form_submit_button("Predict Result")
        
        if submit:
            # This is where your ML model .predict() would go

            st.success(f"Based on your profile, your predicted AI Usage Score is: 4.2 / 5")
