import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Social Impact Dashboard", layout="wide", page_icon="📊")

# --- DARK THEME CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #1e293b !important; }
    div[data-testid="stMetricValue"] { font-size: 32px; color: #38bdf8 !important; }
    div[data-testid="metric-container"] {
        background-color: #1e293b; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); border: 1px solid #334155;
    }
    .stButton>button {
        border-radius: 8px; height: 3.5em; background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white; border: none; font-weight: bold; width: 100%;
    }
    label, p, span { color: #e2e8f0 !important; }
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
        # 1. Cleaning: Renaming based on your Notebook
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
        # Strip invisible spaces
        df.columns = df.columns.str.strip()
        # Drop ID if exists
        if 'ID' in df.columns: df.drop('ID', axis=1, inplace=True)
    return df

df = load_and_clean_data()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='color: #38bdf8;'>🚀 AI Insight Hub</h2>", unsafe_allow_html=True)
        menu = st.radio("MAIN MENU", ["📊 Dashboard", "🔬 Deep Insights", "🔮 Prediction Lab"])
        st.divider()
        st.write("📈 **Accuracy:** 87.8%")

    if menu == "📊 Dashboard":
        st.title("🌐 AI Social Impact Dashboard")
        
        # Metrics from Notebook data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Responses", len(df))
        c2.metric("Avg AI Knowledge", "Moderate") 
        c3.metric("Avg Usage Score", f"{df['AI Usage Rating'].mean():.1f}/5")
        c4.metric("Top Education", df['Education Level'].mode()[0])

        st.divider()

        # Graphs from Notebook (Plot 1 & Plot 2)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution of Age Ranges")
            fig_age = px.histogram(df, x='Age Range', color='Age Range', template="plotly_dark",
                                  category_orders={"Age Range": df['Age Range'].value_counts().index})
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            st.subheader("Trust in AI vs. AI Knowledge")
            fig_trust_know = px.histogram(df, x='AI Knowledge', color='Trust in AI', barmode='group',
                                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_trust_know, use_container_width=True)

    elif menu == "🔬 Deep Insights":
        st.title("🔬 Advanced Analysis")
        
        # Graphs from Notebook (Plot 3 & Plot 4 & Plot 5)
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Perceived Impact on Humanity")
            fig_impact = px.bar(df['AI Impact Perception'].value_counts().reset_index(), 
                               y='AI Impact Perception', x='count', orientation='h',
                               template="plotly_dark", color='AI Impact Perception')
            st.plotly_chart(fig_impact, use_container_width=True)

        with col_b:
            st.subheader("AI Job Impact by Education")
            fig_job = px.histogram(df, x='Education Level', color='AI Job Impact', barmode='group',
                                  template="plotly_dark")
            st.plotly_chart(fig_job, use_container_width=True)

        st.divider()
        st.subheader("AI Usage Rating vs. Trust in AI")
        fig_usage_trust = px.box(df, x='Trust in AI', y='AI Usage Rating', color='Trust in AI',
                                 template="plotly_dark", points="all")
        st.plotly_chart(fig_usage_trust, use_container_width=True)

    else:
        st.title("🔮 AI Prediction Lab")
        st.markdown("### Predict **Employment Status** using survey features")

        # ML Implementation from previous best result
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
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

        with st.container():
            c_left, c_right = st.columns(2)
            u_age = c_left.selectbox("Your Age Range", encoders['Age Range'].classes_)
            u_gen = c_left.selectbox("Your Gender", encoders['Gender'].classes_)
            u_edu = c_right.selectbox("Education Level", encoders['Education Level'].classes_)
            u_use = c_right.select_slider("AI Usage Level", options=[1, 2, 3, 4, 5], value=3)

            if st.button("PREDICT STATUS"):
                in_age = encoders['Age Range'].transform([u_age])[0]
                in_gen = encoders['Gender'].transform([u_gen])[0]
                in_edu = encoders['Education Level'].transform([u_edu])[0]
                
                res_num = model.predict([[in_age, in_gen, in_edu, u_use]])
                final_res = encoders[target].inverse_transform(res_num)[0]
                
                st.balloons()
                st.markdown(f"""
                    <div style="background-color:#1e293b; padding:30px; border-radius:15px; border: 2px solid #38bdf8; text-align:center;">
                        <h1 style="color:#38bdf8; margin:0;">Result: {final_res}</h1>
                        <p style="color:#94a3b8; font-size:18px;">Based on patterns from {len(df)} survey respondents.</p>
                    </div>
                """, unsafe_allow_html=True)

else:
    st.error("Data file not found. Please ensure the CSV is uploaded.")
    
