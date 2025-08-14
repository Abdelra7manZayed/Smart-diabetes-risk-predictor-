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

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn plotly numpy
```

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd diabetes-risk-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   - Place your trained `rf_model.pkl` file in the project root directory
   - The model should be a trained Random Forest classifier

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

## 📁 Project Structure

```
diabetes-risk-predictor/
│
├── app.py                 # Main Streamlit application
├── rf_model.pkl          # Trained Random Forest model (required)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── assets/              # Optional: images, additional resources
```

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

## 🛠️ Customization

### Styling
- Modify CSS in the `st.markdown()` sections for custom appearance
- Adjust color schemes in the style definitions
- Customize chart colors in Plotly configurations

### Features
- Add new input parameters by updating the sidebar and encoding sections
- Modify risk thresholds in the `get_risk_interpretation()` function
- Extend analytics with additional visualizations

### Model Integration
- Replace `rf_model.pkl` with your trained model
- Ensure feature compatibility with existing code
- Update feature names in the importance analysis section

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: Model file 'rf_model.pkl' not found
```
- **Solution**: Ensure `rf_model.pkl` is in the project root directory

**2. Import errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
- **Solution**: Install required packages using `pip install -r requirements.txt`

**3. Prediction errors**
```
Error during prediction: ...
```
- **Solution**: Verify model compatibility and input data format

**4. Visualization issues**
- **Solution**: Update Plotly version: `pip install --upgrade plotly`

## 📊 Requirements.txt

```txt
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.11.0
plotly>=5.15.0
numpy>=1.24.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [Your GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Healthcare professionals for domain expertise
- Scikit-learn community for machine learning tools
- Streamlit team for the amazing web framework
- Plotly for interactive visualizations

---

**⭐ Star this repository if you found it helpful!**

For questions, issues, or contributions, please open an issue on GitHub.
