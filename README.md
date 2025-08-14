# 🩺 Diabetes Risk Predictor

An AI-powered web application that assesses diabetes risk using machine learning. Built with Streamlit and powered by a Random Forest classifier, this tool provides instant risk predictions with interactive visualizations and personalized health recommendations.

## ✨ Features

### 🎯 **Core Functionality**
- **Instant Risk Assessment**: Real-time diabetes risk prediction
- **Interactive Dashboard**: Professional medical-grade interface
- **Risk Categorization**: Low, Moderate, and High-risk classifications
- **Probability Analysis**: Detailed percentage breakdowns with confidence metrics

### 📊 **Advanced Analytics**
- **Feature Importance Analysis**: See which factors contribute most to your risk
- **Health Comparison Charts**: Compare your metrics against population data
- **BMI Classification**: WHO standard weight categories with visual indicators
- **Blood Sugar Analysis**: Normal, pre-diabetes, and diabetes range comparisons

### 🎨 **User Experience**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Visualizations**: Plotly-powered charts and gauges
- **Input Validation**: Real-time feedback for unusual health values
- **Educational Content**: Learn about diabetes risk factors and prevention

### 🔬 **Health Insights**
- **Personalized Recommendations**: Tailored advice based on risk level
- **Risk Factor Summary**: Identification of specific health concerns
- **Medical Disclaimer**: Professional guidance on tool limitations
  
## 🔧 Usage

### Input Parameters
The application requires the following health information:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| **Gender** | Selection | Female/Male | Biological gender |
| **Age** | Slider | 1-120 years | Current age |
| **Hypertension** | Selection | Yes/No | High blood pressure diagnosis |
| **Heart Disease** | Selection | Yes/No | History of heart conditions |
| **Smoking History** | Selection | never/former/current/No Info | Smoking status |
| **BMI** | Slider | 10.0-50.0 | Body Mass Index (kg/m²) |
| **HbA1c Level** | Slider | 3.0-15.0% | Average blood sugar (2-3 months) |
| **Blood Glucose** | Slider | 50-400 mg/dL | Current blood glucose level |

### Getting Predictions
1. **Fill in health information** in the sidebar
2. **Click "Analyze My Diabetes Risk"** for instant prediction
3. **View results** including:
   - Risk level (Low/Moderate/High)
   - Probability percentage
   - Personalized recommendations
   - Feature importance analysis

4. **Optional**: Click "Show Health Analytics" for detailed charts and comparisons

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 8 clinical and demographic factors
- **Output**: Binary classification (Diabetes risk: Yes/No) with probability scores
- **Performance**: Model accuracy depends on training data quality

### Model Requirements
- Must be saved as `rf_model.pkl` using pickle
- Should be trained on features matching the input parameters
- Expected feature order: `[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]`

## ⚠️ Important Disclaimers

### Medical Disclaimer
**This application is for educational and screening purposes only.**
- ❌ **NOT a substitute** for professional medical advice
- ❌ **NOT intended for diagnosis** or treatment decisions
- ❌ **NOT validated** for clinical use

### Recommendations
- ✅ Always consult healthcare professionals for medical decisions
- ✅ Use as a preliminary screening tool only
- ✅ Regular medical check-ups remain essential
- ✅ Discuss results with your doctor


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Abdelrahman Gamal Zayed ** - *Initial work* - [Your GitHub](https://github.com/Abdelra7manZayed)

## 🙏 Acknowledgments

- Healthcare professionals for domain expertise
- Scikit-learn community for machine learning tools
- Streamlit team for the amazing web framework
- Plotly for interactive visualizations

---

**⭐ Star this repository if you found it helpful!**

For questions, issues, or contributions, please open an issue on GitHub.
