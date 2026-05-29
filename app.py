import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Suppress FutureWarning
pd.options.future.no_silent_downcasting = True

# Page configuration
st.set_page_config(
    page_title="AI Student Success Ecosystem",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🎓 AI Student Success Ecosystem</div>', unsafe_allow_html=True)
st.markdown("### Interactive Placement Prediction Dashboard")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("college_student_placement_dataset.csv")
    df.drop("College_ID", axis=1, inplace=True)
    df["Internship_Experience"] = df["Internship_Experience"].replace({"Yes": 1, "No": 0}).infer_objects(copy=False)
    df["Placement"] = df["Placement"].replace({"Yes": 1, "No": 0}).infer_objects(copy=False)
    return df

df = load_data()

# Train model and get accuracy
@st.cache_data
def train_model():
    df_prep = df.copy()
    x = df_prep.drop("Placement", axis=1)
    y = df_prep["Placement"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    
    predictions = rf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return rf, accuracy, x.columns

rf_model, accuracy, feature_names = train_model()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Overview", "Predict Placement", "Model Performance", "Success Insights"]
)

if page == "Home":
    st.markdown("## Welcome to the AI Student Success Ecosystem")
    st.markdown("""
    This dashboard uses machine learning to predict college student placement success 
    and provide personalized recommendations for skill improvement.
    
    ### Key Features:
    - 📊 **Data Overview**: Explore the dataset and understand key features
    - 🔮 **Placement Prediction**: Predict placement chances for individual students
    - 📈 **Model Performance**: View model accuracy and performance metrics
    - 💡 **Success Insights**: Get personalized skill gap analysis and recommendations
    """)
    
    st.info("""
    **How to Use:**
    1. Go to **Predict Placement** to enter student details
    2. Get instant placement probability and predictions
    3. Receive personalized skill gap analysis
    4. Get career path recommendations
    """)

elif page == "Data Overview":
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Information")
        st.write(f"**Total Students:** {len(df)}")
        st.write(f"**Total Features:** {len(df.columns) - 1}")
        st.write(f"**Placed Students:** {df['Placement'].sum()} ({df['Placement'].mean()*100:.1f}%)")
        st.write(f"**Non-Placed Students:** {len(df) - df['Placement'].sum()} ({(1-df['Placement'].mean())*100:.1f}%)")
    
    with col2:
        st.markdown("### Feature Descriptions")
        st.write("""
        - **IQ**: Intelligence Quotient score
        - **Prev_Sem_Result**: Previous semester result/CGPA
        - **CGPA**: Cumulative Grade Point Average
        - **Academic_Performance**: Academic test score (percentage)
        - **Internship_Experience**: Has internship (Yes/No)
        - **Extra_Curricular_Score**: Extra-curricular activities score
        - **Communication_Skills**: Communication skills rating (1-10)
        - **Projects_Completed**: Number of projects completed
        """)
    
    st.markdown("### Dataset Preview")
    st.dataframe(df.head())
    
    st.markdown("### Feature Distribution")
    
    numeric_cols = ['CGPA', 'Academic_Performance', 'Extra_Curricular_Score', 'Communication_Skills']
    fig = px.box(df, y=numeric_cols, title="Distribution of Key Features")
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.pie(values=[df['Placement'].sum(), len(df) - df['Placement'].sum()],
                  names=['Placed', 'Not Placed'],
                  title='Placement Distribution')
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict Placement":
    st.markdown("## 🔮 Predict Student Placement")
    st.markdown("### Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        iq = st.number_input("IQ Score", min_value=70, max_value=145, value=110, step=1)
        prev_sem = st.number_input("Previous Semester Result (CGPA)", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.4, step=0.1)
        academic = st.number_input("Academic Performance (%)", min_value=0, max_value=100, value=85, step=1)
    
    with col2:
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        extra = st.number_input("Extra-curricular Score", min_value=0, max_value=100, value=70, step=1)
        communication = st.number_input("Communication Skills (1-10)", min_value=1, max_value=10, value=7, step=1)
        projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=4, step=1)
    
    internship_num = 1 if internship == "Yes" else 0
    
    if st.button("🔍 Predict Placement", type="primary"):
        # Prepare student data with correct column order
        student_data = pd.DataFrame([[
            iq, prev_sem, cgpa, academic, internship_num, extra, communication, projects
        ]], columns=feature_names)
        
        # Predict
        result = rf_model.predict(student_data)
        probability = rf_model.predict_proba(student_data)
        
        st.markdown("### 📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Placement Probability", f"{probability[0][1]*100:.1f}%")
            st.metric("Prediction", "✅ Placed" if result[0] == 1 else "❌ Not Placed")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability[0][1]*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Placement Chance"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ffebee"},
                        {'range': [30, 60], 'color': "#fff3e0"},
                        {'range': [60, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Success Score
        score = (cgpa * 10 + academic * 0.4 + communication * 5 + projects * 5 + prev_sem * 5)
        st.markdown(f"### 🎯 Student Success Score: **{score:.1f}**")
        
        # Skill Gap Analysis
        st.markdown("### 📋 Skill Gap Analysis")
        
        if communication < 5:
            st.warning("⚠️ **Improve communication skills**")
        if projects < 3:
            st.warning("⚠️ **Work on more projects**")
        if internship_num == 0:
            st.warning("⚠️ **Try getting internship experience**")
        if cgpa > 8:
            st.success("✅ **Academic performance is very good**")
        
        # Learning Recommendations
        st.markdown("### 📚 Recommended Learning")
        
        if communication < 5:
            st.info("📖 **Communication Skills Course**")
        if projects < 3:
            st.info("📖 **Build more Python and ML projects**")
        if cgpa < 7:
            st.info("📖 **Focus on academics and study planning**")
        
        # Career Recommendation
        st.markdown("### 💼 Suggested Career Path")
        
        if projects >= 4 and iq > 100:
            st.success("🚀 **Data Science / Software Development**")
        elif communication > 7:
            st.success("🚀 **Business Analyst**")
        else:
            st.success("🚀 **Web Development**")

elif page == "Model Performance":
    st.markdown("## 📈 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    with col2:
        st.metric("Training Size", f"{int(len(df) * 0.8)} samples")
    
    with col3:
        st.metric("Testing Size", f"{int(len(df) * 0.2)} samples")
    
    with col4:
        st.metric("Features", len(feature_names))
    
    st.markdown("### Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', 
                 orientation='h', title='Feature Importance in Placement Prediction')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Confusion Matrix")
    
    df_prep = df.copy()
    x = df_prep.drop("Placement", axis=1)
    y = df_prep["Placement"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    predictions = rf_model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    title='Confusion Matrix',
                    x=['Not Placed', 'Placed'],
                    y=['Not Placed', 'Placed'])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Success Insights":
    st.markdown("## 💡 Success Insights & Recommendations")
    
    st.markdown("### Key Success Factors")
    
    st.write("""
    Based on the dataset analysis, these factors significantly impact placement:
    """)
    
    tips = [
        ("📚", "Maintain CGPA above 8.0 for better opportunities"),
        ("💬", "Develop strong communication skills (aim for 7+ rating)"),
        ("🛠️", "Complete at least 3-4 substantial projects"),
        ("💼", "Gain internship experience during college"),
        ("📊", "Participate in extra-curricular activities"),
        ("🧠", "Continuously improve technical skills")
    ]
    
    for icon, tip in tips:
        st.markdown(f"{icon} **{tip}**")
    
    st.markdown("### Statistics from Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Students with Internship")
        intern_placed = df[df['Internship_Experience'] == 1]['Placement'].mean() * 100
        st.metric("Placement Rate", f"{intern_placed:.1f}%")
    
    with col2:
        st.markdown("#### Students without Internship")
        no_intern_placed = df[df['Internship_Experience'] == 0]['Placement'].mean() * 100
        st.metric("Placement Rate", f"{no_intern_placed:.1f}%")
    
    st.markdown("### Correlation Heatmap")
    
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>🎓 AI Student Success Ecosystem | Built with Streamlit & Machine Learning</p>
    <p>Random Forest Classifier | Accuracy: {accuracy*100:.2f}%</p>
</div>
""", unsafe_allow_html=True)

# streamlit run app.py