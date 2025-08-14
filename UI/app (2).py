import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin: 0;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton button {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .risk-high {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2em;
    }
    .risk-moderate {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2em;
    }
    .risk-low {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2em;
    }
    .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
@st.cache_data
def load_model():
    """Load the trained model with caching for better performance"""
    try:
        with open("rf_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'rf_model.pkl' not found. Please ensure the model file is in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def validate_inputs(age, bmi, hba1c_level, blood_glucose_level):
    """Validate user inputs and return warnings if needed"""
    warnings = []
    
    if age < 1 or age > 120:
        warnings.append("⚠️ Age seems unusual. Please verify.")
    
    if bmi < 15 or bmi > 50:
        warnings.append("⚠️ BMI is outside typical range (15-50).")
    
    if hba1c_level < 4 or hba1c_level > 15:
        warnings.append("⚠️ HbA1c level seems unusual. Normal range is typically 4-6%.")
    
    if blood_glucose_level < 70 or blood_glucose_level > 400:
        warnings.append("⚠️ Blood glucose level is outside typical range.")
    
    return warnings

def get_bmi_category(bmi):
    """Return BMI category based on WHO standards"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal weight", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

def get_risk_interpretation(probability):
    """Provide detailed risk interpretation"""
    risk_score = probability * 100
    
    if risk_score >= 70:
        return {
            "level": "High Risk",
            "color": "🔴",
            "description": "Strong indicators suggest elevated diabetes risk. Immediate medical consultation recommended.",
            "recommendations": [
                "Schedule immediate appointment with healthcare provider",
                "Consider glucose tolerance test",
                "Review diet and exercise habits",
                "Monitor blood sugar regularly"
            ]
        }
    elif risk_score >= 30:
        return {
            "level": "Moderate Risk",
            "color": "🟠",
            "description": "Some risk factors present. Preventive measures and monitoring recommended.",
            "recommendations": [
                "Regular health check-ups",
                "Maintain healthy weight",
                "Stay physically active",
                "Monitor diet and sugar intake"
            ]
        }
    else:
        return {
            "level": "Low Risk",
            "color": "🟢",
            "description": "Current indicators suggest lower diabetes risk. Maintain healthy lifestyle.",
            "recommendations": [
                "Continue healthy lifestyle",
                "Regular annual check-ups",
                "Stay physically active",
                "Maintain balanced diet"
            ]
        }

# ---------------------------
# LOAD MODEL
# ---------------------------
try:
    rf_model = load_model()
except:
    # Create a dummy model for demonstration if file not found
    st.warning("Using dummy model for demonstration. Please provide 'rf_model.pkl' for actual predictions.")
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Create dummy training data
    dummy_X = np.random.random((100, 8))
    dummy_y = np.random.randint(0, 2, 100)
    rf_model.fit(dummy_X, dummy_y)

# ---------------------------
# SIDEBAR FOR INPUTS
# ---------------------------
with st.sidebar:
    st.header("📋 Health Information")
    st.markdown("Fill in your health details for accurate prediction:")
    
    # Personal Information
    st.subheader("👤 Personal Details")
    gender = st.selectbox("Gender", ["Female", "Male"], help="Biological gender")
    age = st.slider("Age (years)", 1, 120, 30, help="Your current age")
    
    # Medical History
    st.subheader("🏥 Medical History")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"], 
                               help="Have you been diagnosed with high blood pressure?")
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], 
                                help="History of heart-related conditions")
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "No Info"],
        help="Your smoking status"
    )
    
    # Physical Measurements
    st.subheader("📏 Physical Measurements")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, 0.1, 
                   help="Body Mass Index (kg/m²)")
    
    # Calculate and display BMI category
    bmi_category, bmi_icon = get_bmi_category(bmi)
    st.markdown(f"**BMI Category:** {bmi_icon} {bmi_category}")
    
    # Laboratory Tests
    st.subheader("🧪 Laboratory Results")
    hba1c_level = st.slider("HbA1c Level (%)", 3.0, 15.0, 5.5, 0.1,
                           help="Hemoglobin A1C - average blood sugar over 2-3 months")
    blood_glucose_level = st.slider("Blood Glucose (mg/dL)", 50, 400, 100, 1,
                                   help="Current blood glucose level")
    
    # Validation warnings
    validation_warnings = validate_inputs(age, bmi, hba1c_level, blood_glucose_level)
    if validation_warnings:
        st.warning("\n".join(validation_warnings))

# ---------------------------
# MAIN CONTENT
# ---------------------------
st.title("🩺 AI-Powered Diabetes Risk Assessment")
st.markdown("""
<div style='background: linear-gradient(45deg, #3498db, #2980b9); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
    <h3 style='margin: 0; color: white;'>Advanced Machine Learning Diabetes Risk Predictor</h3>
    <p style='margin: 0.5rem 0 0 0; color: white;'>
        Get instant, AI-powered assessment of your diabetes risk based on clinical indicators and lifestyle factors.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# ENCODING AND PREDICTION
# ---------------------------
# Encoding mappings
gender_map = {"Female": 0, "Male": 1}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
smoking_map = {"No Info": 0, "current": 1, "never": 2, "former": 3}

# Create input dataframe
input_df = pd.DataFrame({
    "gender": [gender_map[gender]],
    "age": [age],
    "hypertension": [hypertension_map[hypertension]],
    "heart_disease": [heart_disease_map[heart_disease]],
    "smoking_history": [smoking_map[smoking_history]],
    "bmi": [bmi],
    "HbA1c_level": [hba1c_level],
    "blood_glucose_level": [blood_glucose_level]
})

# ---------------------------
# PREDICTION SECTION
# ---------------------------
col_pred, col_analyze = st.columns([1, 1])

with col_pred:
    if st.button("🔍 Analyze My Diabetes Risk", use_container_width=True):
        st.session_state.prediction_made = True

with col_analyze:
    if st.button("📊 Show Health Analytics", use_container_width=True):
        st.session_state.show_analytics = True

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False

# ---------------------------
# PREDICTION RESULTS
# ---------------------------
if st.session_state.prediction_made:
    try:
        pred = rf_model.predict(input_df)[0]
        proba = rf_model.predict_proba(input_df)[0]
        diabetes_probability = proba[1]
        
        # Get risk interpretation
        risk_info = get_risk_interpretation(diabetes_probability)
        
        st.markdown("---")
        st.header("🎯 Risk Assessment Results")
        
        # Main results in columns
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.subheader("Primary Assessment")
            
            # Risk level display
            if risk_info["level"] == "High Risk":
                st.markdown(f"""
                <div class="risk-high">
                    {risk_info["color"]} <strong>HIGH RISK</strong><br>
                    Diabetes Risk: {diabetes_probability*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            elif risk_info["level"] == "Moderate Risk":
                st.markdown(f"""
                <div class="risk-moderate">
                    {risk_info["color"]} <strong>MODERATE RISK</strong><br>
                    Diabetes Risk: {diabetes_probability*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    {risk_info["color"]} <strong>LOW RISK</strong><br>
                    Diabetes Risk: {diabetes_probability*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Interpretation:** {risk_info['description']}")
        
        with col2:
            # Probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = diabetes_probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.subheader("Recommendations")
            for i, rec in enumerate(risk_info["recommendations"], 1):
                st.markdown(f"{i}. {rec}")
        
        # Detailed probability breakdown
        st.subheader("📈 Probability Analysis")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric(
                label="No Diabetes Probability",
                value=f"{proba[0]*100:.1f}%",
                delta=f"{(proba[0]-0.5)*100:.1f}% vs baseline" if proba[0] != 0.5 else None
            )
        
        with prob_col2:
            st.metric(
                label="Diabetes Risk Probability", 
                value=f"{proba[1]*100:.1f}%",
                delta=f"{(proba[1]-0.5)*100:.1f}% vs baseline" if proba[1] != 0.5 else None
            )
        
        # Feature importance visualization
        st.subheader("🔍 Key Risk Factors Analysis")
        
        feature_names = ["Gender", "Age", "Hypertension", "Heart Disease", 
                        "Smoking History", "BMI", "HbA1c Level", "Blood Glucose"]
        importance = rf_model.feature_importances_
        
        # Create interactive importance chart
        fig_importance = px.bar(
            x=importance,
            y=feature_names,
            orientation='h',
            title="Feature Importance in Risk Assessment",
            labels={'x': 'Importance Score', 'y': 'Health Factor'},
            color=importance,
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please ensure all inputs are valid and the model is properly loaded.")

# ---------------------------
# HEALTH ANALYTICS SECTION
# ---------------------------
if st.session_state.show_analytics:
    st.markdown("---")
    st.header("📊 Health Analytics Dashboard")
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI comparison chart
        st.subheader("BMI Analysis")
        
        # Generate sample population data for comparison
        np.random.seed(42)
        ages_sample = np.random.normal(45, 15, 1000)
        ages_sample = np.clip(ages_sample, 18, 80)
        bmis_sample = np.random.normal(26, 4, 1000)
        bmis_sample = np.clip(bmis_sample, 18, 40)
        
        fig_bmi = px.scatter(
            x=ages_sample, y=bmis_sample,
            opacity=0.3,
            title="Your BMI vs Population",
            labels={'x': 'Age (years)', 'y': 'BMI'},
            color_discrete_sequence=['lightblue']
        )
        
        # Add user's data point
        fig_bmi.add_scatter(
            x=[age], y=[bmi],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Your Data'
        )
        
        # Add BMI category lines
        fig_bmi.add_hline(y=18.5, line_dash="dash", line_color="blue", 
                         annotation_text="Underweight")
        fig_bmi.add_hline(y=25, line_dash="dash", line_color="green", 
                         annotation_text="Normal")
        fig_bmi.add_hline(y=30, line_dash="dash", line_color="orange", 
                         annotation_text="Overweight")
        
        st.plotly_chart(fig_bmi, use_container_width=True)
    
    with col2:
        # HbA1c and Glucose levels
        st.subheader("Blood Sugar Analysis")
        
        # Create ranges chart
        ranges_data = {
            'Measure': ['HbA1c Level', 'Blood Glucose'],
            'Your Value': [hba1c_level, blood_glucose_level],
            'Normal Range Min': [4.0, 70],
            'Normal Range Max': [5.7, 100],
            'Pre-diabetes Min': [5.7, 100],
            'Pre-diabetes Max': [6.4, 125],
            'Diabetes Threshold': [6.5, 126]
        }
        
        fig_ranges = go.Figure()
        
        # Add ranges
        fig_ranges.add_trace(go.Bar(
            name='Normal Range',
            x=['HbA1c (%)', 'Glucose (mg/dL)'],
            y=[5.7-4.0, 100-70],
            base=[4.0, 70],
            marker_color='green',
            opacity=0.3
        ))
        
        fig_ranges.add_trace(go.Bar(
            name='Pre-diabetes Range',
            x=['HbA1c (%)', 'Glucose (mg/dL)'],
            y=[6.4-5.7, 125-100],
            base=[5.7, 100],
            marker_color='orange',
            opacity=0.3
        ))
        
        # Add user values
        fig_ranges.add_trace(go.Scatter(
            x=['HbA1c (%)', 'Glucose (mg/dL)'],
            y=[hba1c_level, blood_glucose_level],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Your Values'
        ))
        
        fig_ranges.update_layout(
            title="Your Blood Sugar Levels vs Normal Ranges",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig_ranges, use_container_width=True)
    
    # Risk factors summary
    st.subheader("Risk Factors Summary")
    risk_factors = []
    
    if age > 45:
        risk_factors.append("• Age over 45 increases diabetes risk")
    if bmi >= 25:
        risk_factors.append(f"• BMI of {bmi:.1f} indicates overweight/obesity")
    if hba1c_level >= 5.7:
        risk_factors.append(f"• HbA1c level of {hba1c_level}% suggests elevated blood sugar")
    if blood_glucose_level >= 100:
        risk_factors.append(f"• Blood glucose of {blood_glucose_level} mg/dL is above normal")
    if hypertension == "Yes":
        risk_factors.append("• Hypertension is a diabetes risk factor")
    if heart_disease == "Yes":
        risk_factors.append("• Heart disease correlates with diabetes risk")
    if smoking_history == "current":
        risk_factors.append("• Current smoking increases diabetes risk")
    
    if risk_factors:
        st.warning("**Identified Risk Factors:**\n" + "\n".join(risk_factors))
    else:
        st.success("✅ No major risk factors identified based on current inputs!")

# ---------------------------
# EDUCATIONAL CONTENT
# ---------------------------
st.markdown("---")
st.header("📚 Learn About Diabetes Risk")

with st.expander("🔬 Understanding the Risk Factors"):
    st.markdown("""
    **Key Diabetes Risk Factors:**
    
    - **Age**: Risk increases after 45, especially after 65
    - **BMI**: Higher body weight increases insulin resistance
    - **HbA1c**: Shows average blood sugar over 2-3 months
    - **Blood Glucose**: Immediate indicator of blood sugar levels
    - **Hypertension**: Often co-occurs with diabetes
    - **Heart Disease**: Shares risk factors with diabetes
    - **Smoking**: Increases insulin resistance and complications
    """)

with st.expander("📊 About This AI Model"):
    st.markdown("""
    **Model Information:**
    
    - **Algorithm**: Random Forest Classifier
    - **Features**: 8 clinical and demographic factors
    - **Purpose**: Risk assessment and early detection support
    - **Limitations**: Not a replacement for medical diagnosis
    
    **How it Works:**
    The model analyzes patterns in your health data compared to thousands of cases
    to estimate diabetes risk probability.
    """)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin: 2rem 0;'>
    <h4>⚠️ Important Medical Disclaimer</h4>
    <p>This AI tool is designed for <strong>educational and screening purposes only</strong>. 
    It is <strong>not a substitute for professional medical advice</strong>, diagnosis, or treatment.</p>
    
    <p><strong>Always consult with qualified healthcare providers</strong> for:</p>
    <ul style='text-align: left; max-width: 600px; margin: 0 auto;'>
        <li>Medical diagnosis and treatment decisions</li>
        <li>Interpretation of test results</li>
        <li>Medication management</li>
        <li>Personalized health recommendations</li>
    </ul>
    
    <p style='margin-top: 1rem;'>
        <strong>Accuracy Note:</strong> Predictions are based on statistical models and may not reflect individual cases.
        Regular medical check-ups remain the gold standard for health monitoring.
    </p>
</div>
""", unsafe_allow_html=True)

# Add reset button
if st.button("🔄 Reset Analysis", help="Clear all results and start over"):
    st.session_state.prediction_made = False
    st.session_state.show_analytics = False
    st.rerun()
