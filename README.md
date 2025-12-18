# Heart-Attack-risk-Predictor

A machine learning web application that predicts cardiovascular risk levels based on personal, medical, and lifestyle information. Built with Streamlit and Scikit-learn, this app provides real-time predictions along with visual probability distributions for different risk levels.

Features:-
Real-time risk prediction – Predicts Low, Medium, or High cardiovascular risk.
Interactive visualizations – Probability distribution of each risk level displayed via Plotly bar charts.
BMI auto-calculation – Calculates BMI automatically from height and weight inputs.
Height conversion – Accepts height in feet & inches and converts it to centimeters.
User-friendly interface – Clean, responsive design with a soft pink theme.
Handles categorical and numerical inputs – Includes age, smoking, alcohol, exercise, and other medical factors.

How to Run:-
Clone the repository:
git clone https://github.com/PrachiTawar/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
The app will open in your browser at http://localhost:8501.
Live:- 
https://huggingface.co/spaces/Prach-404/Heart-Attack-Risk-Predictor

Model:-
Algorithm: Logistic Regression
Features: [Replace with actual number, e.g., 20 features including age, gender, blood pressure, cholesterol, BMI, etc.]
Target: Cardiovascular risk level (Low, Medium, High)
Preprocessing: Label encoding for categorical features, StandardScaler for numerical features

Dataset:-
The model was trained on a heart disease dataset including medical and lifestyle features, enhanced with diet quality information.

Disclaimer:-
⚠️ Informational Only: This app is for educational purposes and not a medical diagnosis tool. Predictions are based on historical data and may not reflect actual health conditions. Always consult a healthcare professional for medical advice.

Technologies Used
Python – Data processing and backend
Streamlit – Web application framework
Scikit-learn – Machine learning model
Pandas & NumPy – Data manipulation
Plotly – Interactive visualizations
