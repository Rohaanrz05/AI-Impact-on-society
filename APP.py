import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="AI Impact Dashboard", layout="wide", page_icon="🤖")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'The impact of artificial intelligence on society.csv'
    # These are the most common encodings for CSVs that cause errors
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None

df = load_data()

# --- APP LAYOUT ---
if df is not None:
    st.title("🌐 The Impact of AI on Society")
    st.markdown("This dashboard analyzes public perception, trust, and usage frequency of AI across different demographics.")

    # --- SIDEBAR ---
    st.sidebar.header("📊 Global Filters")
    
    # Clean up column names for easier access (optional but helpful)
    # Filter by Gender
    gender_options = df['Gender'].unique().tolist()
    selected_gender = st.sidebar.multiselect("Select Gender", gender_options, default=gender_options)

    # Filter by Age
    age_options = df['Age'].unique().tolist()
    selected_age = st.sidebar.multiselect("Select Age Groups", age_options, default=age_options)

    # Apply Filters
    mask = (df['Gender'].isin(selected_gender)) & (df['Age'].isin(selected_age))
    filtered_df = df[mask]

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", len(filtered_df))
    col2.metric("Unique Age Groups", len(selected_age))
    col3.metric("Avg. Trust (Approx)", "Moderate") # Place holder logic

    # --- TABS FOR VISUALIZATION ---
    tab1, tab2, tab3 = st.tabs(["📈 Demographic Trends", "🛡️ Trust & Ethics", "🔍 Raw Data"])

    with tab1:
        st.subheader("How different groups engage with AI")
        c1, c2 = st.columns(2)
        
        with c1:
            # Pie Chart for Education
            fig_pie = px.pie(filtered_df, names='Education', title='Education Level of Respondents',
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            # Bar Chart for AI Usage Frequency
            usage_col = 'How often do you use products that use AI? (Example: ChatGPT, Netflix, Siri, etc.)'
            if usage_col in df.columns:
                fig_bar = px.histogram(filtered_df, x=usage_col, color='Gender', 
                                     title="Usage Frequency by Gender",
                                     barmode='group')
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Perception of AI Trustworthiness")
        trust_col = 'Do you trust the artificial intelligence systems?'
        if trust_col in df.columns:
            fig_trust = px.box(filtered_df, x='Age', y=trust_col, color='Gender',
                               title="Trust Distribution across Age and Gender")
            st.plotly_chart(fig_trust, use_container_width=True)
        else:
            st.warning("Trust column not found. Please check CSV headers.")

    with tab3:
        st.subheader("Data Explorer")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered CSV", data=csv, file_name="filtered_ai_data.csv", mime="text/csv")

else:
    st.error("🚨 **Error:** Could not load the CSV file. Please ensure 'The impact of artificial intelligence on society.csv' is in the same folder as this script.")
    st.info("Technical Tip: The error usually stems from the file encoding. We tried UTF-8 and ISO-8859-1 without success.")
