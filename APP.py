import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="AI Impact Dashboard", layout="wide")

# Function to load data with encoding fallback
@st.cache_data
def load_data():
    file_path = 'The impact of artificial intelligence on society.csv'
    # List of encodings to try
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    return None

# Load the dataset
df = load_data()

if df is not None:
    st.title("🌐 The Impact of AI on Society")
    st.markdown("Exploring public perception, trust, and usage of Artificial Intelligence.")

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Data")
    
    # Filter by Gender
    gender_list = df['Gender'].unique().tolist()
    selected_gender = st.sidebar.multiselect("Select Gender", gender_list, default=gender_list)

    # Filter by Age
    age_list = df['Age'].unique().tolist()
    selected_age = st.sidebar.multiselect("Select Age Group", age_list, default=age_list)

    # Apply filters
    filtered_df = df[(df['Gender'].isin(selected_gender)) & (df['Age'].isin(selected_age))]

    # --- MAIN DASHBOARD TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Sentiment Analysis", "🔮 AI Predictor"])

    with tab1:
        st.subheader("General Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Total Respondents:**", len(filtered_df))
            # Pie chart for Education
            fig_edu = px.pie(filtered_df, names='Education', title='Education Levels')
            st.plotly_chart(fig_edu, use_container_width=True)

        with col2:
            # Bar chart for Trust levels
            # Note: Ensure the column name matches exactly as per your CSV
            trust_col = 'Do you trust the artificial intelligence systems?'
            if trust_col in filtered_df.columns:
                fig_trust = px.histogram(filtered_df, x=trust_col, color='Gender', 
                                         title='Trust in AI Systems', barmode='group')
                st.plotly_chart(fig_trust, use_container_width=True)

    with tab2:
        st.subheader("Usage & Sentiment")
        # Scatter plot or box plot for Usage frequency
        usage_col = 'How often do you use products that use AI? (Example: ChatGPT, Netflix, Siri, etc.)'
        if usage_col in filtered_df.columns:
            fig_usage = px.box(filtered_df, x='Education', y=usage_col, 
                               points="all", title="AI Usage Frequency by Education")
            st.plotly_chart(fig_usage, use_container_width=True)

    with tab3:
        st.subheader("Predictive Insights")
        st.info("This section can be used to predict user sentiment based on demographics.")
        st.write("Current data shows that the most frequent users are in the following age groups:")
        st.write(filtered_df['Age'].value_counts())

else:
    st.error("Failed to load the dataset. Please check if the file name and encoding are correct.")
