from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import shap
import logging
import sys
from pathlib import Path

# Sets up the project root path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Sets up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vars for models and preprocessor
models = {}
explainers = {}
feature_columns = []
region_lookup = {}
city_lookup = {}
preprocessor = None

async def load_models():
    """
    Loads all trained models and SHAP explainers on startup
    """
    global models, explainers, feature_columns, preprocessor, region_lookup, city_lookup
    
    try:
        # Gets the absolute path to the project root
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "results" / "models"
        
        logger.info(f"Looking for models in: {models_dir.absolute()}")
        
        # Lists all pickle files in the models directory
        if models_dir.exists():
            pkl_files = list(models_dir.glob("*.pkl"))
            logger.info(f"Found pickle files: {[f.name for f in pkl_files]}")
        else:
            logger.error(f"Models directory does not exist: {models_dir}")
            return
        # Loads your three models (using absolute paths)
        model_files = {
            'logistic': models_dir / 'logistic_regression_best.pkl',
            'xgboost': models_dir / 'xgboost_best.pkl', 
            'svm': models_dir / 'svm_rbf_best.pkl'
        }
        
        models = {}
        for name, path in model_files.items():
            if path.exists():
                models[name] = joblib.load(path)
                logger.info(f"Loaded {name} model from {path}")
            else:
                logger.warning(f"Model file not found: {path}")
        
        # Loads SHAP explainers
        explainer_files = {
            'logistic': models_dir / 'logistic_explainer.pkl',
            'xgboost': models_dir / 'xgboost_explainer.pkl',
            'svm': models_dir / 'svm_explainer.pkl'
        }
        
        explainers = {}
        for name, path in explainer_files.items():
            if path.exists():
                explainers[name] = joblib.load(path)
                logger.info(f"Loaded {name} explainer from {path}")
            else:
                logger.warning(f"Explainer file not found: {path}")
        
        # Loads feature columns
        feature_columns_path = models_dir / 'feature_columns.pkl'
        if feature_columns_path.exists():
            feature_columns = joblib.load(feature_columns_path)
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        else:
            logger.warning(f"Feature columns file not found: {feature_columns_path}")
            # Uses default from preprocessor
            feature_columns = []
        
        # Loads the preprocessor
        preprocessor_path = models_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor successfully")
        else:
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor required for API operation")
        
        # Loads the dropdown data and creates lookup dictionaries
        data_dir = project_root / "data" / "processed"
        
        # Loads the exact region and exact city names
        regions_csv = data_dir / "unique_regions.csv"
        cities_csv = data_dir / "unique_cities.csv"
        
        if regions_csv.exists() and cities_csv.exists():
            regions_df = pd.read_csv(regions_csv)
            cities_df = pd.read_csv(cities_csv)
            
            # Creates a case INSENSITIVE lookup dictionaries
            region_lookup = {region.lower(): region for region in regions_df['region']}
            city_lookup = {city.lower(): city for city in cities_df['city']}
            
            logger.info(f"Loaded {len(region_lookup)} regions and {len(city_lookup)} cities for lookup")
        else:
            logger.warning("Dropdown CSV files not found. Case sensitivity may cause issues...")
            region_lookup = {}
            city_lookup = {}
        
        logger.info("All models and preprocessor loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

def normalize_region(user_input: str) -> str:
    """Converts user input to exact model format"""
    if not user_input:
        return user_input
    return region_lookup.get(user_input.lower(), user_input)

def normalize_city(user_input: str) -> str:
    """Converts user input to exact model format"""  
    if not user_input:
        return user_input
    return city_lookup.get(user_input.lower(), user_input)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_models()
    yield
    # Shutdown
    pass

# FastAPI app w/ lifespan
app = FastAPI(
    title="Startup Success Prediction API",
    description="Predict startup acquisition success using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configures appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantics models for request/response - UPDATED TO MATCH MODEL INPUTS
class StartupFeatures(BaseModel):
    country_code: str
    region: str
    city: str
    category_list: str  # Will be processed into binary category features
    founded_year: int
    
    @field_validator('founded_year')
    @classmethod
    def validate_founded_year(cls, v):
        if v < 1995 or v > 2015:  # Updated to match your training data range
            raise ValueError('Founded year must be between 1995 and 2015')
        return v

class PredictionResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    success_probability: float
    prediction: int
    model_used: str
    confidence: str

class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    feature_importance: Dict[str, float]
    top_factors: List[Dict[str, Any]]

def preprocess_features(features: StartupFeatures) -> np.ndarray:
    """
    Preprocessess input features to match training data format
    """
    try:
        # Normalizes region and city to exact model format
        normalized_region = normalize_region(features.region)
        normalized_city = normalize_city(features.city)
        
        # Converts Pydantic model to dictionary with correct field names
        feature_dict = {
            'country_code': features.country_code,
            'region': normalized_region,  
            'city': normalized_city,     
            'category_list': features.category_list,
            'founded_year': features.founded_year
        }
        
        # Uses the loaded preprocessor to transform the data
        processed_features = preprocessor.transform_single(feature_dict)
        return processed_features
        
    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Startup Success Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "explainers_loaded": len(explainers),
        "preprocessor_loaded": preprocessor is not None,
        "expected_features": [
            "country_code", "region", "city", "category_list", "founded_year"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_success(features: StartupFeatures):
    """
    Predicts startup success probability
    """
    try:
        if not models:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Preprocess features
        X = preprocess_features(features)
        X = X.reshape(1, -1)  # Ensure 2D array for prediction
        
        # Uses XGBoost as default model
        model_name = 'xgboost'
        if model_name not in models:
            model_name = list(models.keys())[0]  # Use first available model
        
        model = models[model_name]
        
        # Gets prediction
        probability = model.predict_proba(X)[0][1]  # Probability of success (class 1)
        prediction = int(probability > 0.5)
        
        # Determines confidence level
        confidence = "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.1 else "low"
        
        return PredictionResponse(
            success_probability=float(probability),
            prediction=prediction,
            model_used=model_name,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/explain", response_model=ExplanationResponse)
async def predict_with_explanation(features: StartupFeatures):
    """
    Predict startup success with SHAP explanations
    """
    try:
        if not models or not explainers:
            raise HTTPException(status_code=503, detail="Models or explainers not loaded")
        
        # Gets basic prediction
        prediction_response = await predict_success(features)
        
        # Preprocess features for SHAP
        X = preprocess_features(features)
        X = X.reshape(1, -1)
        
        # Uses same model as prediction
        model_name = prediction_response.model_used
        explainer = explainers.get(model_name)
        
        if not explainer:
            raise HTTPException(status_code=503, detail=f"Explainer for {model_name} not available")
        
        # Gets SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handles different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Creates feature importance dictionary
        if len(feature_columns) == len(shap_values[0]):
            feature_importance = dict(zip(feature_columns, shap_values[0]))
        else:
            # Fallback to generic names if feature names don't match
            feature_importance = {f"feature_{i}": val for i, val in enumerate(shap_values[0])}
        
        # Gets top 5 most important features
        top_factors = []
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for feature_name, importance in sorted_features:
            top_factors.append({
                "feature": feature_name,
                "importance": float(importance),
                "impact": "positive" if importance > 0 else "negative"
            })
        
        return ExplanationResponse(
            prediction=prediction_response,
            feature_importance=feature_importance,
            top_factors=top_factors
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/models")
async def list_models():
    """
    Lists available models and their status
    """
    return {
        "available_models": list(models.keys()),
        "available_explainers": list(explainers.keys()),
        "feature_columns_count": len(feature_columns),
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/categories")
async def get_available_categories():
    """
    Returns the list of available categories for the frontend
    """
    categories = [
        'software', 'mobile', 'social', 'media', 'web', 'e-commerce', 
        'biotechnology', 'curated', 'health', 'advertising', 'games', 
        'enterprise', 'technology', 'marketing', 'analytics'
    ]
    return {"categories": categories}

@app.get("/regions")
async def get_available_regions():
    """
    Returns the list of available regions for the frontend dropdown
    """
    regions = sorted(list(region_lookup.values()))
    return {"regions": regions}

@app.get("/cities") 
async def get_available_cities():
    """
    Returns the list of available cities for the frontend dropdown
    """
    cities = sorted(list(city_lookup.values()))
    return {"cities": cities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)