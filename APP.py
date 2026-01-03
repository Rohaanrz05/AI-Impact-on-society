import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Impact Dashboard", layout="wide")

@st.cache_data
def load_data():
    file_path = 'The impact of artificial intelligence on society.csv'
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            # Strip whitespace from column names to prevent matching errors
            df.columns = df.columns.str.strip()
            return df
        except:
            continue
    return None

df = load_data()

if df is not None:
    st.title("🌐 AI Society Impact Dashboard")
    
    # --- AUTOMATIC COLUMN LOCATOR ---
    # This finds columns even if the names are slightly different
    gender_col = next((c for c in df.columns if 'gender' in c.lower()), df.columns[1])
    age_col = next((c for c in df.columns if 'age' in c.lower()), df.columns[0])
    trust_col = next((c for c in df.columns if 'trust' in c.lower()), None)
    usage_col = next((c for c in df.columns if 'use' in c.lower() or 'often' in c.lower()), None)

    # --- SIDEBAR ---
    st.sidebar.header("Filters")
    selected_gender = st.sidebar.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())
    selected_age = st.sidebar.multiselect("Age Group", df[age_col].unique(), default=df[age_col].unique())

    filtered_df = df[df[gender_col].isin(selected_gender) & df[age_col].isin(selected_age)]

    # --- DASHBOARD ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        fig1 = px.pie(filtered_df, names=gender_col, title="Gender Distribution", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("AI Sentiment")
        if trust_col:
            fig2 = px.histogram(filtered_df, x=trust_col, color=gender_col, barmode='group', title="Trust Levels")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("Trust data column not found.")

    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(10))

else:
    st.error("Could not load data. Please check the file name on GitHub.")
