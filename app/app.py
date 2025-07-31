from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from src.data_preprocessing import preprocess_single_input, StartupDataProcessor
import joblib
import pandas as pd
import numpy as np
import shap
import logging
from pathlib import Path

# Configures logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vars for models and explainers
models = {}
explainers = {}
feature_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Startup Success Predictor API", 
    version="1.0.0",
    lifespan=lifespan
)

# Enables CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL specification neede for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class StartupFeatures(BaseModel):
    country_code: str
    region: str
    city: str
    category_list: str
    market: str
    founded_year: int
    funding_total_usd: Optional[float] = 0.0
    
    @field_validator('founded_year')
    @classmethod
    def validate_founded_year(cls, v):
        if not 1990 <= v <= 2015:
            raise ValueError('Founded year must be between 1990 and 2015')
        return v

class PredictionResponse(BaseModel):
    success_probability: float
    confidence_interval: Dict[str, float]
    predicted_class: str
    model_used: str

class ExplanationResponse(BaseModel):
    feature_importance: Dict[str, float]
    shap_values: List[float]
    feature_names: List[str]
    base_value: float

async def load_models():
    """
    Loads all trained models and SHAP explainers on startup
    """
    global models, explainers, feature_columns
    
    try:
        # Loads three models
        model_files = {
            'logistic': 'models/logistic_regression_best.pkl',
            'svm': 'models/svm_rbf_best.pkl', 
            'xgboost': 'models/xgboost_best.pkl'
        }
        
        for model_name, model_path in model_files.items():
            if Path(model_path).exists():
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model successfully")
                
                # Loads corresponding SHAP explainer if exists
                explainer_path = f'models/{model_name}_explainer.pkl'
                if Path(explainer_path).exists():
                    explainers[model_name] = joblib.load(explainer_path)
                    logger.info(f"Loaded {model_name} SHAP explainer")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Loads feature columns
        feature_columns_path = 'models/feature_columns.pkl'
        if Path(feature_columns_path).exists():
            feature_columns = joblib.load(feature_columns_path)
            logger.info("Loaded feature columns")
        
        if not models:
            raise Exception("No models loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "explainers_loaded": list(explainers.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_startup_success(features: StartupFeatures, model_type: str = "xgboost"):
    """
    Predicts startup success probability
    """
    
    if model_type not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_type}' not available. Available models: {list(models.keys())}"
        )
    
    try:
        # Preprocess features
        processed_features = preprocess_features(features)
        
        # Makes prediction
        model = models[model_type]
        prediction_proba = model.predict_proba(processed_features)[0]
        success_probability = float(prediction_proba[1])  # Assuming binary classification
        
        # Calculates confidence interval 
        confidence_interval = {
            "lower": max(0.0, success_probability - 0.1),
            "upper": min(1.0, success_probability + 0.1)
        }
        
        predicted_class = "Success" if success_probability > 0.5 else "Failure"
        
        return PredictionResponse(
            success_probability=success_probability,
            confidence_interval=confidence_interval,
            predicted_class=predicted_class,
            model_used=model_type
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(features: StartupFeatures, model_type: str = "xgboost"):
    """
    Get SHAP explanation for a prediction
    """
    
    if model_type not in explainers:
        raise HTTPException(
            status_code=400,
            detail=f"SHAP explainer for '{model_type}' not available"
        )
    
    try:
        # Preprocess features
        processed_features = preprocess_features(features)
        
        # Gets SHAP values
        explainer = explainers[model_type]
        shap_values = explainer.shap_values(processed_features)
        
        # Handles different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, takes positive class
        
        # Creates feature importance dictionary
        feature_importance = {}
        if feature_columns is not None:
            for i, feature_name in enumerate(feature_columns):
                if i < len(shap_values[0]):
                    feature_importance[feature_name] = float(shap_values[0][i])
        
        return ExplanationResponse(
            feature_importance=feature_importance,
            shap_values=shap_values[0].tolist(),
            feature_names=feature_columns or [],
            base_value=float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

def preprocess_features(features: StartupFeatures) -> np.ndarray:
    """
    Preprocess input features to match training data format
    Replicates preprocessing pipeline from notebooks
    """
    # TODO: Implement the same preprocessing logic from notebooks
    # This is a placeholder
    
    # Creates a DataFrame with the input features
    feature_dict = {
        'country_code': features.country_code,
        'region': features.region,
        'city': features.city,
        'category_list': features.category_list,
        'market': features.market,
        'founded_year': features.founded_year,
        'funding_total_usd': features.funding_total_usd
    }
    
    df = pd.DataFrame([feature_dict])
    
    # Apply the same preprocessing steps from notebooks:
    # 1. Geographic density encoding
    # 2. Industry one-hot encoding  
    # 3. Feature scaling
    # 4. Any other transformations
    
    # For now, return a dummy array - will replace this with actual preprocessing pipeline
    if feature_columns is not None:
        dummy_features = np.zeros((1, len(feature_columns)))
        return dummy_features
    else:
        # Returns basic features as placeholder
        return np.array([[
            hash(features.country_code) % 100,
            hash(features.region) % 100, 
            hash(features.city) % 100,
            features.founded_year,
            features.funding_total_usd or 0.0
        ]])

def preprocess_features(features: StartupFeatures) -> np.ndarray:
    """
    Preprocess input features to match training data format
    """
    try:
        # Converts Pydantic model to dictionary
        features_dict = {
            'country_code': features.country_code,
            'region': features.region,
            'city': features.city,
            'category_list': features.category_list,
            'market': features.market,
            'founded_year': features.founded_year,
            'funding_total_usd': features.funding_total_usd or 0.0
        }
        
        # Uses the preprocessing pipeline
        processed_features = preprocess_single_input(features_dict, "models/")
        return processed_features
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        # Fallback to basic preprocessing if the pipeline fails
        return np.array([[
            hash(features.country_code) % 100,
            hash(features.region) % 100, 
            hash(features.city) % 100,
            features.founded_year,
            features.funding_total_usd or 0.0
        ]])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)