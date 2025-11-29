from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import pickle
import shap
from typing import Optional
import httpx
from pydantic import BaseModel
import os

app = FastAPI(title="Type 2 Diabetes Prediction System")

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Global variables
model = None
explainer = None

# Biomistral API configuration
BIOMISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
BIOMISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "Nggm1Rpja2QShMb7VJuAwr3zErJvRn3J")  

class PredictionInput(BaseModel):
    age: float
    gender: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    smoking_history: str
    hypertension: int
    heart_disease: int


@app.on_event("startup")
async def startup_event():
    """Load model and initialize SHAP explainer on startup"""
    global model, explainer
    try:
        # Load the model
        model = pickle.load(open('ensemble_model.pkl', 'rb'))
        print("Model loaded successfully")
        
        # Initialize SHAP explainer
        initialize_explainer()
        print("SHAP explainer initialized successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise


def initialize_explainer():
    """Initialize SHAP explainer with background data"""
    global explainer
    try:
        # Create a small background dataset with typical values
        background_data = pd.DataFrame({
            'age': [40, 50, 60, 30, 70],
            'hypertension': [0, 1, 0, 0, 1],
            'heart_disease': [0, 0, 1, 0, 1],
            'bmi': [25, 30, 28, 22, 32],
            'HbA1c_level': [5.5, 6.0, 5.8, 5.2, 6.5],
            'blood_glucose_level': [100, 120, 110, 90, 140],
            'smoking_history_No Info': [0, 0, 0, 1, 0],
            'smoking_history_current': [0, 1, 0, 0, 0],
            'smoking_history_ever': [0, 0, 0, 0, 0],
            'smoking_history_former': [0, 0, 1, 0, 0],
            'smoking_history_never': [1, 0, 0, 0, 1],
            'smoking_history_not current': [0, 0, 0, 0, 0],
            'gender_Female': [1, 0, 1, 0, 1],
            'gender_Male': [0, 1, 0, 1, 0],
            'gender_Other': [0, 0, 0, 0, 0]
        })

        # Feature column order expected by the model
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
            'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
        ]

        # Try TreeExplainer first, fallback to KernelExplainer
        try:
            explainer = shap.TreeExplainer(model, background_data)
            print("SHAP TreeExplainer initialized successfully")
        except Exception:
            def predict_proba_pos(x):
                df_x = pd.DataFrame(x, columns=feature_columns)
                return model.predict_proba(df_x)[:, 1]

            explainer = shap.KernelExplainer(predict_proba_pos, background_data[feature_columns].values)
            print("SHAP KernelExplainer initialized (fallback)")

    except Exception as e:
        print(f"Error initializing SHAP explainer: {str(e)}")
        raise




async def get_biomistral_explanation(shap_data: dict, patient_data: dict, probability: float) -> str:
    """
    Use Biomistral LLM to generate natural language explanation of SHAP values
    """
    # Check if API key is set
    if not BIOMISTRAL_API_KEY == "Nggm1Rpja2QShMb7VJuAwr3zErJvRn3J":
        print("Warning: Mistral API key not configured. Using fallback explanation.")
        return generate_fallback_explanation(shap_data, patient_data, probability)
    
    try:
        # Prepare the prompt for Biomistral
        prompt = f"""You are a medical AI assistant. Provide a clear diabetes risk explanation.

Patient Data:
- Age: {patient_data['age']} years
- Gender: {patient_data['gender']}
- BMI: {patient_data['bmi']}
- HbA1c Level: {patient_data['HbA1c_level']}%
- Blood Glucose: {patient_data['blood_glucose_level']} mg/dL
- Smoking: {patient_data['smoking_history']}
- Hypertension: {'Yes' if patient_data['hypertension'] == 1 else 'No'}
- Heart Disease: {'Yes' if patient_data['heart_disease'] == 1 else 'No'}

Prediction: {'Diabetic' if probability >= 50 else 'Non-diabetic'} ({probability:.0f}% probability)

Technical SHAP Values (contribution to prediction):
{format_shap_factors_for_llm(shap_data)}

Provide a medical explanation in this format:

"This patient {'presents high' if probability >= 50 else 'shows low'} diabetes risk. [Start with the most important factor - 
explain what the value is, compare it to clinical thresholds (normal/prediabetes/diabetes ranges), and state what it indicates].
 [Then discuss 2-5 other key factors, explaining their values and significance]. [For high risk: emphasize clinical evaluation needed. 
 For low risk: emphasize maintaining healthy behaviors]. These factors collectively {'indicate this patient likely has Type 2 Diabetes and requires comprehensive clinical evaluation' if probability >= 50 else 'suggest minimal diabetes risk at this time'}."

Keep it professional, medically accurate, and around 100-150 words."""

     
        model_names = [
          
            "open-mistral-7b",           # Open Mistral
           
        ]
        
        # Call Mistral API
        async with httpx.AsyncClient(timeout=30.0) as client:
            for model_name in model_names:
                
                try:
                    response = await client.post(
                        BIOMISTRAL_API_URL,
                        headers={
                            "Authorization": f"Bearer {BIOMISTRAL_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model_name,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a medical AI assistant. Provide clear, concise explanations of diabetes risk using medical terminology and clinical thresholds."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.7,
                            "max_tokens": 400
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        explanation = result['choices'][0]['message']['content']
                        print(f"Successfully used model: {model_name}")
                        return explanation
                    else:
                        print(f"Model {model_name} failed: {response.status_code}")
                        continue
                        
                except Exception as model_error:
                    print(f"Error with model {model_name}: {str(model_error)}")
                    continue
            
            # If the  Mistral model fails
            print(f"All Mistral models failed. Status: {response.status_code}, Response: {response.text}")
            return generate_fallback_explanation(shap_data, patient_data, probability)
                
    except Exception as e:
        print(f"Error calling Mistral API: {str(e)}")
        return generate_fallback_explanation(shap_data, patient_data, probability)


def format_shap_factors_for_llm(shap_data: dict) -> str:
    """Format SHAP factors for LLM prompt with raw SHAP values"""
    if not shap_data or 'top_factors' not in shap_data:
        return "No factor data available."
    
    formatted = []
    for factor in shap_data['top_factors'][:5]:
        # Format: HbA1c (+0.48), Glucose (+0.35)
        shap_val = factor.get('shap_value', 0)
        sign = '+' if shap_val >= 0 else ''
        formatted.append(f"{factor['feature']} ({sign}{shap_val:.2f})")
    
    return ", ".join(formatted)


def generate_fallback_explanation(shap_data: dict, patient_data: dict, probability: float) -> str:
    """Generate explanation matching the example format with SHAP values"""
    
    # Getting patient values
    hba1c = float(patient_data.get('HbA1c_level', 0))
    glucose = float(patient_data.get('blood_glucose_level', 0))
    bmi = float(patient_data.get('bmi', 0))
    age = int(float(patient_data.get('age', 0)))
    
    # Determining prediction
    prediction = "Diabetic" if probability >= 50 else "Non-diabetic"
    risk_level = "high" if probability >= 50 else "low"
    
    # Start explanation
    explanation = f"This patient {'presents high' if probability >= 50 else 'shows low'} diabetes risk. "
    
    # Get top factors with SHAP values
    top_factors = shap_data.get('top_factors', [])[:4] if shap_data else []
    
    # Analyze primary factor (usually HbA1c or Glucose)
    if top_factors:
        primary = top_factors[0]
        
        if 'HbA1c' in primary['feature']:
            if hba1c >= 6.5:
                explanation += f"The primary concern is their HbA1c level of {hba1c}%, which exceeds the diabetic threshold (≥6.5%) and strongly indicates diabetes. "
            elif hba1c >= 5.7:
                explanation += f"Their HbA1c level of {hba1c}% is in the prediabetes range (5.7-6.4%), indicating elevated blood sugar control over the past 2-3 months. "
            else:
                explanation += f"Their HbA1c level of {hba1c}% is within the normal range (below 5.7%), indicating healthy long-term blood glucose control. "
        
        elif 'Glucose' in primary['feature'] or 'Blood Glucose' in primary['feature']:
            if glucose >= 126:
                explanation += f"The primary concern is their fasting blood glucose of {glucose:.0f} mg/dL, which is in the diabetic range (≥126 mg/dL) and strongly indicates diabetes. "
            elif glucose >= 100:
                explanation += f"Their fasting blood glucose of {glucose:.0f} mg/dL is elevated (100-125 mg/dL range), indicating impaired fasting glucose. "
            else:
                explanation += f"Their fasting glucose of {glucose:.0f} mg/dL is normal (below 100 mg/dL). "
        
        elif 'BMI' in primary['feature']:
            if bmi >= 30:
                explanation += f"The primary concern is their BMI of {bmi}, which indicates obesity, a well-established diabetes risk factor. "
            elif bmi >= 25:
                explanation += f"Their BMI of {bmi} indicates overweight status, which moderately increases diabetes risk. "
            else:
                explanation += f"Their BMI of {bmi} falls in the healthy weight range. "
    
    # Add supporting factors
    if len(top_factors) > 1:
        secondary_factors = []
        
        for factor in top_factors[1:4]:
            if 'HbA1c' in factor['feature'] and 'HbA1c' not in (top_factors[0]['feature'] if top_factors else ''):
                if hba1c >= 6.5:
                    secondary_factors.append(f"HbA1c of {hba1c}% is in the diabetic range")
                elif hba1c >= 5.7:
                    secondary_factors.append(f"HbA1c of {hba1c}% indicates prediabetes")
                else:
                    secondary_factors.append(f"HbA1c of {hba1c}% is normal")
            
            elif ('Glucose' in factor['feature'] or 'Blood Glucose' in factor['feature']) and 'Glucose' not in (top_factors[0]['feature'] if top_factors else ''):
                if glucose >= 126:
                    secondary_factors.append(f"fasting blood glucose of {glucose:.0f} mg/dL is also in the diabetic range (≥126 mg/dL), providing supporting evidence")
                elif glucose >= 100:
                    secondary_factors.append(f"fasting glucose of {glucose:.0f} mg/dL is elevated")
                else:
                    secondary_factors.append(f"fasting glucose of {glucose:.0f} mg/dL is normal")
            
            elif 'BMI' in factor['feature'] and 'BMI' not in (top_factors[0]['feature'] if top_factors else ''):
                if bmi >= 30:
                    secondary_factors.append(f"BMI of {bmi} indicates obesity")
                elif bmi >= 25:
                    secondary_factors.append(f"BMI of {bmi} indicates overweight status")
                else:
                    secondary_factors.append(f"BMI of {bmi} is in the healthy range")
            
            elif 'Age' in factor['feature']:
                if age >= 45:
                    secondary_factors.append(f"at age {age}, they are in an age range with elevated diabetes prevalence")
                else:
                    secondary_factors.append(f"their age ({age}) is favorable")
            
            elif 'Hypertension' in factor['feature']:
                if patient_data.get('hypertension') == 1:
                    secondary_factors.append("presence of hypertension increases risk")
                else:
                    secondary_factors.append("no hypertension is favorable")
            
            elif 'Heart Disease' in factor['feature']:
                if patient_data.get('heart_disease') == 1:
                    secondary_factors.append("history of heart disease compounds risk")
                else:
                    secondary_factors.append("no heart disease is favorable")
            
            elif 'current' in factor['feature'].lower():
                secondary_factors.append("current smoking status significantly increases risk")
            elif 'former' in factor['feature'].lower():
                secondary_factors.append("former smoking history shows some residual risk")
            elif 'never' in factor['feature'].lower():
                secondary_factors.append("never smoking is protective")
        
        if secondary_factors:
            if len(secondary_factors) == 1:
                explanation += f"Their {secondary_factors[0]}. "
            elif len(secondary_factors) == 2:
                explanation += f"Their {secondary_factors[0]}, and their {secondary_factors[1]}. "
            else:
                explanation += f"Additionally, their {secondary_factors[0]}, their {secondary_factors[1]}, and their {secondary_factors[2]}. "
    
    # Conclusion
    if probability >= 50:
        explanation += "These factors collectively indicate this patient likely has Type 2 Diabetes and requires comprehensive clinical evaluation."
    else:
        explanation += "These favorable biomarkers suggest minimal diabetes risk at this time."
    
    return explanation


def encode_features(data: dict):
    """Encode categorical features to match the model's expected format"""
    encoded_data = {
        'age': float(data['age']),
        'hypertension': int(data['hypertension']),
        'heart_disease': int(data['heart_disease']),
        'bmi': float(data['bmi']),
        'HbA1c_level': float(data['HbA1c_level']),
        'blood_glucose_level': float(data['blood_glucose_level']),
        'smoking_history_No Info': 0,
        'smoking_history_current': 0,
        'smoking_history_ever': 0,
        'smoking_history_former': 0,
        'smoking_history_never': 0,
        'smoking_history_not current': 0,
        'gender_Female': 0,
        'gender_Male': 0,
        'gender_Other': 0
    }
    
    # Set the appropriate smoking history column to 1
    smoking_col = f"smoking_history_{data['smoking_history']}"
    if smoking_col in encoded_data:
        encoded_data[smoking_col] = 1
    
    # Set the appropriate gender column to 1
    gender_col = f"gender_{data['gender']}"
    if gender_col in encoded_data:
        encoded_data[gender_col] = 1
    
    return encoded_data


def get_feature_importance_explanation(df, shap_values):
    """
    Get top factors influencing the prediction with readable names using SHAP values
    """
    # Handle SHAP values based on output format
    print(f"SHAP values type: {type(shap_values)}")
    
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
    elif isinstance(shap_values, list):
        if len(shap_values) == 2:
            if isinstance(shap_values[1], np.ndarray):
                shap_vals = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
            else:
                shap_vals = shap_values[1]
        else:
            shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
    else:
        shap_vals = np.array(shap_values).flatten()
    
    print(f"Processed SHAP values shape: {shap_vals.shape}")
    print(f"Sample SHAP values: {shap_vals[:5]}")
    
    # Feature name mapping
    feature_names = {
        'age': 'Age',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'bmi': 'BMI',
        'HbA1c_level': 'HbA1c Level',
        'blood_glucose_level': 'Blood Glucose Level',
        'smoking_history_No Info': 'Smoking History: No Info',
        'smoking_history_current': 'Smoking History: Current',
        'smoking_history_ever': 'Smoking History: Ever',
        'smoking_history_former': 'Smoking History: Former',
        'smoking_history_never': 'Smoking History: Never',
        'smoking_history_not current': 'Smoking History: Not Current',
        'gender_Female': 'Gender: Female',
        'gender_Male': 'Gender: Male',
        'gender_Other': 'Gender: Other'
    }
    
    # Get feature names and their SHAP values
    features = df.columns.tolist()
    feature_values = df.iloc[0].values
    
    # Combine features with their SHAP values
    importance_data = []
    for i, feature in enumerate(features):
        shap_value = float(shap_vals[i]) if i < len(shap_vals) else 0.0
        
        importance_data.append({
            'feature': feature_names.get(feature, feature),
            'shap_value': abs(shap_value),
            'actual_shap': shap_value,
            'raw_shap': round(shap_value, 4),  # Raw SHAP value with 4 decimals
            'value': feature_values[i],
            'impact': 'Increases' if shap_value > 0 else 'Decreases'
        })
    
    # Sort by absolute SHAP value
    importance_data.sort(key=lambda x: x['shap_value'], reverse=True)
    
    print(f"Top 5 factors before filtering:")
    for item in importance_data[:5]:
        print(f"  {item['feature']}: SHAP={item['raw_shap']}, Value={item['value']}")
    
    # Get top 5 factors
    top_factors = []
    for item in importance_data[:5]:
        # Include all top 5 factors, even if SHAP value is small
        if item['value'] in [0, 1]:
            value_display = 'Yes' if item['value'] == 1 else 'No'
        else:
            value_display = f"{item['value']:.2f}"
        
        # Get basic interpretation
        interpretation = get_basic_interpretation(item['feature'], value_display, item['actual_shap'])
        
        top_factors.append({
            'feature': item['feature'],
            'value': value_display,
            'impact': item['impact'],
            'importance': round(item['shap_value'] * 100, 2),
            'shap_value': item['raw_shap'],  # Include the raw SHAP value
            'interpretation': interpretation
        })
    
    return top_factors


def get_basic_interpretation(feature_name: str, value: str, shap_value: float) -> str:
    """Provide basic interpretation for factors"""
    interpretations = {
        'Age': f"Age is {'an important risk factor' if shap_value > 0 else 'a protective factor'}.",
        'BMI': f"BMI of {value} {'increases' if shap_value > 0 else 'decreases'} diabetes risk.",
        'HbA1c Level': f"HbA1c level {'indicates elevated blood sugar' if shap_value > 0 else 'is within normal range'}.",
        'Blood Glucose Level': f"Blood glucose level {'is elevated' if shap_value > 0 else 'is normal'}.",
        'Hypertension': f"{'Having' if value == 'Yes' else 'Not having'} hypertension affects risk.",
        'Heart Disease': f"{'Having' if value == 'Yes' else 'Not having'} heart disease affects risk.",
    }
    
    return interpretations.get(feature_name, f"This factor {' increases' if shap_value > 0 else 'decreases'} risk.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(
    age: float = Form(...),
    gender: str = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: float = Form(...),
    smoking_history: str = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...)
):
    """Make diabetes prediction with SHAP explanation"""
    try:
        # Prepare data
        data = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'smoking_history': smoking_history,
            'hypertension': hypertension,
            'heart_disease': heart_disease
        }
        
        print("Received data:", data)
        
        # Encode features
        encoded_data = encode_features(data)
        
        # Convert to DataFrame
        feature_columns = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'smoking_history_No Info', 'smoking_history_current',
            'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
            'smoking_history_not current', 'gender_Female', 'gender_Male', 'gender_Other'
        ]
        
        df = pd.DataFrame([encoded_data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        # Calculate SHAP values
        top_factors = []
        try:
            if isinstance(explainer, shap.KernelExplainer):
                shap_values = explainer.shap_values(df.values, nsamples=100)
            else:
                shap_values = explainer.shap_values(df)
            
            top_factors = get_feature_importance_explanation(df, shap_values)
            print(f"Top factors calculated: {len(top_factors)} factors")
        except Exception as shap_error:
            print(f"SHAP calculation error: {str(shap_error)}")
        
        # Calculate risk level
        if probability < 20:
            risk_level = "Low"
            risk_color = "#28a745"
        elif probability < 50:
            risk_level = "Moderate"
            risk_color = "#ffc107"
        else:
            risk_level = "High"
            risk_color = "#dc3545"
        
        # Prepare SHAP data for LLM
        shap_data = {
            'top_factors': top_factors
        }
        
        # Get Biomistral explanation
        llm_explanation = await get_biomistral_explanation(shap_data, data, probability)
        
        return JSONResponse({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'top_factors': top_factors,
            'llm_explanation': llm_explanation
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None, "explainer_loaded": explainer is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    