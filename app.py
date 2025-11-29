import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - SOFT PINK THEME
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #ffe6f2;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffccdd;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #660033 !important;
        font-weight: bold !important;
    }
    
    /* Text */
    p, label, .stMarkdown {
        color: #660033 !important;
    }
    
    /* Input boxes */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #fff0f5 !important;
        color: #660033 !important;
        border: 2px solid #ff99cc !important;
        border-radius: 10px !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #ff99cc !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s !important;
    }
    
    .stButton button:hover {
        background-color: #ff66b3 !important;
        transform: scale(1.05) !important;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #fff0f5 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 2px solid #ff99cc !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: #ffccdd !important;
        color: #660033 !important;
        border-radius: 10px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #cc0066 !important;
        font-size: 2em !important;
    }
    
    /* Divider */
    hr {
        border-color: #ff99cc !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================
@st.cache_resource
def load_model_objects():
    """Load the trained model and preprocessing objects from .pkl files"""
    try:
        # Load all saved .pkl files
        with open("lr_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("target_encoder.pkl", "rb") as f:
            target_encoder = pickle.load(f)

        with open("feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)

        with open("categorical_cols.pkl", "rb") as f:
            categorical_cols = pickle.load(f)

        with open("numerical_cols.pkl", "rb") as f:
            numerical_cols = pickle.load(f)

        # Load original dataset if needed
        df_original = pd.read_csv("heart_dataset_with_diet_quality.csv")
        return {
            'model': model,
            'label_encoders': label_encoders,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'feature_columns': feature_columns,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'df_original': df_original
        }

    except Exception as e:
        st.error(f"Error loading model objects: {str(e)}")
        return None

# ============================================
# HELPER FUNCTIONS
# ============================================
def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def feet_to_cm(feet, inches=0):
    """Convert feet and inches to centimeters"""
    total_inches = (feet * 12) + inches
    cm = total_inches * 2.54
    return round(cm, 2)

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>💗 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Enter patient information to predict cardiovascular risk level</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model objects
    model_objects = load_model_objects()
    
    if model_objects is None:
        st.error("Failed to load model. Please ensure the model is trained in the notebook.")
        return
    
    model = model_objects['model']
    label_encoders = model_objects['label_encoders']
    scaler = model_objects['scaler']
    target_encoder = model_objects['target_encoder']
    feature_columns = model_objects['feature_columns']
    categorical_cols = model_objects['categorical_cols']
    numerical_cols = model_objects['numerical_cols']
    df_original = model_objects['df_original']
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model:** Logistic Regression  
        **Features:** {len(feature_columns)}  
        **Risk Classes:** {len(target_encoder.classes_)}
          
        """)
        
        st.markdown("### 🎯 Risk Classes")
        for cls in target_encoder.classes_:
            st.markdown(f"• {cls}")
    
    # Create input form
    st.markdown("## 📝 Patient Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    user_input = {}
    
    # Special fields for height, weight, BMI
    height_cm = None
    weight_kg = None
    calculated_bmi = None
    
    with col1:
        st.markdown("### Personal & Medical Factors")
        user_name = st.text_input("Name", placeholder="Enter name")
        
        for i, col in enumerate(feature_columns[:len(feature_columns)//2]):
            # Special handling for Age
            if col.lower() == 'age':
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()}",
                    min_value=0,
                    max_value=120,
                    value=int(df_original[col].mean()),
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Alcohol Consumption
            elif 'alcohol' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Smoking
            elif 'smoking' in col.lower() or 'smoke' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Exercise
            elif 'exercise' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Height
            elif col.lower() == 'height' or 'height' in col.lower():
                st.markdown(f"**{col.replace('_', ' ').title()}**")
                height_unit = st.radio("Height Unit", ["Centimeters", "Feet"], key=f"unit_{col}", horizontal=True)
                
                if height_unit == "Centimeters":
                    height_cm = st.number_input(
                        "Height (cm)",
                        min_value=50.0,
                        max_value=250.0,
                        value=170.0,
                        step=0.1,
                        key=f"input_{col}_cm"
                    )
                else:
                    col_ft, col_in = st.columns(2)
                    with col_ft:
                        feet = st.number_input("Feet", min_value=1, max_value=8, value=5, step=1, key=f"input_{col}_ft")
                    with col_in:
                        inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1, key=f"input_{col}_in")
                    height_cm = feet_to_cm(feet, inches)
                    st.info(f"Height: {height_cm} cm")
                
                user_input[col] = height_cm
            
            # Special handling for Weight
            elif col.lower() == 'weight' or 'weight' in col.lower():
                weight_kg = st.number_input(
                    f"{col.replace('_', ' ').title()} (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    value=70.0,
                    step=0.1,
                    key=f"input_{col}"
                )
                user_input[col] = weight_kg
            
            # Special handling for BMI
            elif col.lower() == 'bmi' or col == 'BMI':
                # Calculate BMI if height and weight are available
                if height_cm is not None and weight_kg is not None:
                    calculated_bmi = calculate_bmi(weight_kg, height_cm)
                    st.markdown(f"**{col.replace('_', ' ').title()}** (Auto-calculated)")
                    st.info(f"BMI: {calculated_bmi}")
                    user_input[col] = calculated_bmi
                else:
                    # Fallback if height/weight not yet entered
                    user_input[col] = st.number_input(
                        f"{col.replace('_', ' ').title()} (will be auto-calculated)",
                        min_value=10.0,
                        max_value=60.0,
                        value=float(df_original[col].mean()) if col in df_original.columns else 25.0,
                        disabled=True,
                        key=f"input_{col}"
                    )
            
            elif col in categorical_cols:
                # Get unique values from original dataframe
                unique_values = sorted(df_original[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(
                    f"{col.replace('_', ' ').title()}",
                    options=unique_values,
                    key=f"input_{col}"
                )
            elif col in numerical_cols:
                # Get min, max, mean for default values
                min_val = float(df_original[col].min())
                max_val = float(df_original[col].max())
                mean_val = float(df_original[col].mean())
                
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{col}"
                )
    
    with col2:
        st.markdown("### Additional Factors")
        for i, col in enumerate(feature_columns[len(feature_columns)//2:]):
            # Special handling for Age
            if col.lower() == 'age':
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()}",
                    min_value=0,
                    max_value=120,
                    value=int(df_original[col].mean()),
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Alcohol Consumption
            elif 'alcohol' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Smoking
            elif 'smoking' in col.lower() or 'smoke' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Exercise
            elif 'exercise' in col.lower():
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()} (0-10 scale)",
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    key=f"input_{col}"
                )
            # Special handling for Height
            elif col.lower() == 'height' or 'height' in col.lower():
                st.markdown(f"**{col.replace('_', ' ').title()}**")
                height_unit = st.radio("Height Unit", ["Centimeters", "Feet"], key=f"unit_{col}", horizontal=True)
                
                if height_unit == "Centimeters":
                    height_cm = st.number_input(
                        "Height (cm)",
                        min_value=50.0,
                        max_value=250.0,
                        value=170.0,
                        step=0.1,
                        key=f"input_{col}_cm"
                    )
                else:
                    col_ft, col_in = st.columns(2)
                    with col_ft:
                        feet = st.number_input("Feet", min_value=1, max_value=8, value=5, step=1, key=f"input_{col}_ft")
                    with col_in:
                        inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1, key=f"input_{col}_in")
                    height_cm = feet_to_cm(feet, inches)
                    st.info(f"Height: {height_cm} cm")
                
                user_input[col] = height_cm
            
            # Special handling for Weight
            elif col.lower() == 'weight' or 'weight' in col.lower():
                weight_kg = st.number_input(
                    f"{col.replace('_', ' ').title()} (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    value=70.0,
                    step=0.1,
                    key=f"input_{col}"
                )
                user_input[col] = weight_kg
            
            # Special handling for BMI
            elif col.lower() == 'bmi' or col == 'BMI':
                # Calculate BMI if height and weight are available
                if height_cm is not None and weight_kg is not None:
                    calculated_bmi = calculate_bmi(weight_kg, height_cm)
                    st.markdown(f"**{col.replace('_', ' ').title()}** (Auto-calculated)")
                    st.info(f"BMI: {calculated_bmi}")
                    user_input[col] = calculated_bmi
                else:
                    # Fallback if height/weight not yet entered
                    user_input[col] = st.number_input(
                        f"{col.replace('_', ' ').title()} (will be auto-calculated)",
                        min_value=10.0,
                        max_value=60.0,
                        value=float(df_original[col].mean()) if col in df_original.columns else 25.0,
                        disabled=True,
                        key=f"input_{col}"
                    )
            
            elif col in categorical_cols:
                unique_values = sorted(df_original[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(
                    f"{col.replace('_', ' ').title()}",
                    options=unique_values,
                    key=f"input_{col}"
                )
            elif col in numerical_cols:
                min_val = float(df_original[col].min())
                max_val = float(df_original[col].max())
                mean_val = float(df_original[col].mean())
                
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"input_{col}"
                )
    
    st.markdown("---")
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Risk Level", use_container_width=True)
    
    if predict_button:
        try:
            # Prepare input data
            input_df = pd.DataFrame([user_input])
            
            # Encode categorical features
            for col in categorical_cols:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            
            # Scale numerical features
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Ensure correct feature order
            input_df = input_df[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            predicted_class = target_encoder.inverse_transform([prediction])[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Main prediction
            st.markdown(f"<h2 style='text-align: center; color: #cc0066;'>Predicted Risk Level: {predicted_class}</h2>", unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### 📊 Risk Probability Distribution")
            
            # Create probability chart
            classes = target_encoder.classes_
            prob_df = pd.DataFrame({
                'Risk Level': classes,
                'Probability': probabilities * 100
            })
            
            # Sort by probability
            prob_df = prob_df.sort_values('Probability', ascending=True)
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Risk Level'],
                orientation='h',
                marker=dict(
                    color=prob_df['Probability'],
                    colorscale=[[0, '#ffccdd'], [0.5, '#ff99cc'], [1, '#cc0066']],
                    showscale=False
                ),
                text=[f'{p:.1f}%' for p in prob_df['Probability']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Risk Level Probabilities',
                xaxis_title='Probability (%)',
                yaxis_title='Risk Level',
                plot_bgcolor='#ffe6f2',
                paper_bgcolor='#ffe6f2',
                font=dict(color='#660033', size=12),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            st.markdown("### 📈 Detailed Probabilities")
            cols = st.columns(len(classes))
            for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                with cols[i]:
                    st.metric(
                        label=cls,
                        value=f"{prob*100:.2f}%"
                    )
            
            # Confidence indicator
            max_prob = max(probabilities)
            if max_prob > 0.7:
                confidence = "High"
                color = "#00cc66"
            elif max_prob > 0.5:
                confidence = "Medium"
                color = "#ff9900"
            else:
                confidence = "Low"
                color = "#cc0066"
            
            st.markdown(f"<p style='text-align: center; font-size: 1.2em;'>Prediction Confidence: <strong style='color: {color};'>{confidence}</strong> ({max_prob*100:.1f}%)</p>", unsafe_allow_html=True)
            st.sidebar.markdown("""
            **⚠️ Disclaimer:**  
            This predictor provides **informational results only**.  
            It is **not a medical diagnosis**.  
            Consult a **healthcare professional** for medical advice.
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
          

if __name__ == "__main__":
    main()
