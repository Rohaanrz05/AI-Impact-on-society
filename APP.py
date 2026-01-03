import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Society Dashboard", layout="wide", page_icon="🤖")

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
    # --- SMART COLUMN DETECTOR ---
    age_col = next((c for c in df.columns if 'age' in c.lower()), df.columns[0])
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), df.columns[1])
    edu_col = next((c for c in df.columns if 'education' in c.lower()), df.columns[2])
    trust_col = next((c for c in df.columns if 'trust' in c.lower()), None)

    st.title("🌐 AI Impact & Predictive Analytics")

    tab1, tab2 = st.tabs(["📊 Data Visualization", "🔮 AI Trust Predictor"])

    with tab1:
        # (Your existing visualization code)
        st.subheader("Public Sentiment Overview")
        fig_hist = px.histogram(df, x=trust_col, color=gender_col, barmode='group', title="Trust Levels by Gender")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Can we predict if someone trusts AI?")
        st.write("This model uses Random Forest to predict trust based on demographics.")

        # --- PREPARING DATA FOR ML ---
        # We need to turn text (Male/Female) into numbers (0/1) for the model
        ml_df = df[[age_col, gender_col, edu_col, trust_col]].dropna()
        
        le_dict = {}
        for col in [age_col, gender_col, edu_col, trust_col]:
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col])
            le_dict[col] = le

        # Train the model
        X = ml_df[[age_col, gender_col, edu_col]]
        y = ml_df[trust_col]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # --- USER INPUT FORM ---
        st.divider()
        st.write("### Try it yourself:")
        u_age = st.selectbox("Your Age Group", options=le_dict[age_col].classes_)
        u_gen = st.selectbox("Your Gender", options=le_dict[gender_col].classes_)
        u_edu = st.selectbox("Education Level", options=le_dict[edu_col].classes_)

        if st.button("Predict My Trust Level"):
            # Transform user input to numbers
            in_age = le_dict[age_col].transform([u_age])[0]
            in_gen = le_dict[gender_col].transform([u_gen])[0]
            in_edu = le_dict[edu_col].transform([u_edu])[0]
            
            prediction = model.predict([[in_age, in_gen, in_edu]])
            result = le_dict[trust_col].inverse_transform(prediction)[0]
            
            st.success(f"Based on survey patterns, your predicted trust level is: **{result}**")
            st.balloons()

else:
    st.error("Could not load data. Check GitHub file path.")
