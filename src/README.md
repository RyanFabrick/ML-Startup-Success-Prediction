# *Backend*

## Table of Contents
- [Overview](#overview)
- [Backend Architecture](#backend-architecture)
- [API Endpoints](#api-endpoints)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Installation & Setup](#installation--setup)'
- [Demo GIFs](#demo-gifs)
- [Usage Examples](#usage-examples)
- [Model Integration](#model-integration)
- [Production Deployment](#production-deployment)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)

## Overview

The backend system of this project is a FastAPI based REST API that serves machine learning predictions for startup success using trained XGBoost, Logistic Regression, and SVM models. urrently, the full stack application is powered by the trained XGBoost due to its relative perfromance compared to the other models. It provides real time predictions with SHAP based explanations while maintaining strict adherence to the bias free methodology established in the research notebooks.

**Key Features:**
- **Real time ML Inference**: Serves predictions from XGBoost trained model with sub second response times
- **SHAP Integration**: Provides model interpretability through SHapley Additive exPlanations for individual predictions
- **Production Ready Architecture**: Implements proper error handling, request validation, and health monitoring
- **Bias Free Processing**: Maintains founding time only feature restrictions preventing look ahead bias

## Backend Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                         │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│ │ Prediction  │ │ Explanation │ │ Health      │                 │
│ │ Endpoints   │ │ Endpoints   │ │ Monitoring  │                 │
│ └─────────────┘ └─────────────┘ └─────────────┘                 │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│ │ Pydantic    │ │ CORS        │ │ Async       │                 │
│ │ Validation  │ │ Middleware  │ │ Lifecycle   │                 │
│ └─────────────┘ └─────────────┘ └─────────────┘                 │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                Data Processing Layer                            │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│ │Startup Data │ │Feature      │ │Geographic   │                 │
│ │Processor    │ │Engineering  │ │Normalization│                 │
│ └─────────────┘ └─────────────┘ └─────────────┘                 │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐                                 │
│ │Category     │ │Temporal     │                                 │
│ │Processing   │ │Features     │                                 │
│ └─────────────┘ └─────────────┘                                 │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                    Model Layer                                  │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│ │ XGBoost     │ │ Logistic    │ │ SVM RBF     │                 │
│ │ Model       │ │ Regression  │ │ Model       │                 │
│ │ (.pkl)      │ │ (.pkl)      │ │ (.pkl)      │                 │
│ └─────────────┘ └─────────────┘ └─────────────┘                 │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│ │ SHAP        │ │ Feature     │ │ Preprocessor│                 │
│ │ Explainers  │ │ Columns     │ │ Pipeline    │                 │
│ │ (.pkl)      │ │ (.pkl)      │ │ (.pkl)      │                 │
│ └─────────────┘ └─────────────┘ └─────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Core Prediction Endpoints

#### `POST /predict`
Basic startup success prediction with confidence scoring

**Request Body:**
```json
{
  "country_code": "USA",
  "region": "SF Bay Area", 
  "city": "San Francisco",
  "category_list": "software mobile",
  "founded_year": 2010
}
```

**Response:**
```json
{
  "success_probability": 0.342,
  "prediction": 0,
  "model_used": "xgboost",
  "confidence": "medium"
}
```

#### `POST /predict/explain`
Enhanced prediction with SHAP based feature importance explanations

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "prediction": {
    "success_probability": 0.342,
    "prediction": 0,
    "model_used": "xgboost", 
    "confidence": "medium"
  },
  "feature_importance": {
    "founded_year_std": 0.125,
    "era_recovery": -0.089,
    "is_usa": 0.067,
    "category_software": 0.045,
    "region_startup_density": 0.032
  },
  "top_factors": [
    {
      "feature": "founded_year_std",
      "importance": 0.125,
      "impact": "positive"
    }
  ]
}
```

### Utility Endpoints

#### `GET /health`
System health monitoring and model status verification

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "explainers_loaded": 3,
  "preprocessor_loaded": true,
  "expected_features": [
    "country_code", "region", "city", "category_list", "founded_year"
  ]
}
```

#### `GET /categories`
Available industry categories for frontend dropdown population

**Response:**
```json
{
  "categories": [
    "software", "mobile", "social", "media", "web", 
    "e-commerce", "biotechnology", "curated", "health",
    "advertising", "games", "enterprise", "technology",
    "marketing", "analytics"
  ]
}
```

#### `GET /regions` & `GET /cities`
Geographic location options for user selection interfaces

**Response:**
```json
{
  "regions": ["SF Bay Area", "New York City", "Boston", "Los Angeles", ...],
  "cities": ["San Francisco", "New York", "Boston", "Los Angeles", ...]
}
```

## Data Processing Pipeline

### StartupDataProcessor Class

The core preprocessing engine that transforms raw startup data into model-ready feature vectors

**Key Capabilities:**
- **Geographic Feature Engineering**: Converts regions/cities into 5-tier startup density rankings
- **Industry Categorization**: Processes category strings into binary feature encodings for 15 major sectors
- **Temporal Feature Creation**: Standardizes founding years and assigns economic era classifications
- **Production Optimization**: Handles single record transformation for real-  API inference

**Feature Engineering Process:**

1. **Geographic Processing (3 features)**:
   - `region_startup_density`: 5-tier ranking system (1=highest density, 5=lowest)
   - `city_startup_density`: City level startup concentration ranking
   - `is_usa`: Binary flag for US-based companies (captures ecosystem advantages)

2. **Industry Processing (15 features)**:
   - Binary encoding for major categories: software, mobile, social, media, web, e-commerce, biotechnology, curated, health, advertising, games, enterprise, technology, marketing, analytics
   - Handles multi category assignments and parsing artifact removal

3. **Temporal Processing (4 features)**:
   - `founded_year_std`: Standardized founding year using training data statistics
   - `era_dotcom_era`: Binary flag for 1995-2000 period
   - `era_post_crash`: Binary flag for 2001-2008 period  
   - `era_recovery`: Binary flag for 2009-2015 period

### Data Flow Architecture

```python
# Raw Input (API Request)
{
  "country_code": "USA",
  "region": "SF Bay Area",
  "city": "San Francisco", 
  "category_list": "software mobile",
  "founded_year": 2010
}

# ↓ StartupDataProcessor.transform_single()

# Processed Features (22-dimensional vector)
[
  5,    # region_startup_density (SF Bay Area = tier 5)
  5,    # city_startup_density (San Francisco = tier 5)
  1,    # is_usa (True)
  1,    # category_software (True)
  1,    # category_mobile (True)
  0,    # category_social (False)
  ...   # Additional category flags
  0.35, # founded_year_std ((2010 - 2007.4) / 7.6)
  0,    # era_dotcom_era (False)
  0,    # era_post_crash (False) 
  1     # era_recovery (True)
]

# ↓ Model Inference (XGBoost/Logistic/SVM)

# Prediction Output
{
  "success_probability": 0.342,
  "prediction": 0,
  "confidence": "medium"
}
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Trained ML models in `../results/models/` directory

### Environment Setup

```bash
# Clone repository
git clone https://github.com/RyanFabrick/Startup-Success-Prediction.git
cd Startup-Success-Prediction

# Create virtual environment
python -m venv startup_predictor_env
source startup_predictor_env/bin/activate  # Linux/Mac
# startup_predictor_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Model Artifacts

Ensure these files exist in `../results/models/`:
```
├── models/
│   ├── xgboost_best.pkl          # Trained XGBoost model
│   ├── logistic_regression_best.pkl  # Trained Logistic Regression
│   ├── svm_rbf_best.pkl          # Trained SVM model
│   ├── xgboost_explainer.pkl     # SHAP explainer for XGBoost
│   ├── logistic_explainer.pkl    # SHAP explainer for Logistic
│   ├── svm_explainer.pkl         # SHAP explainer for SVM
│   ├── feature_columns.pkl       # Feature column specifications
│   └── preprocessor.pkl          # Fitted data preprocessor
```

### Launch Server

```bash
# Development server
cd app
python app.py
# Server runs on http://localhost:8000

# Production server (alternative)
cd app
uvicorn app:app --host 0.0.0.0 --port 8000

# Production with Gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
## Demo GIFs

![backenddemogif1](https://github.com/user-attachments/assets/7b7e65ae-92fb-4ba0-910e-1a88204e87ab)

![demobackendgif2](https://github.com/user-attachments/assets/10ef554f-b1b8-487a-af1b-51e70dd1291e)

## Usage Examples

### Basic Prediction

```python
import requests

# Startup data
startup_data = {
    "country_code": "USA",
    "region": "SF Bay Area",
    "city": "San Francisco", 
    "category_list": "software enterprise",
    "founded_year": 2010
}

# Get prediction
response = requests.post(
    "http://localhost:8000/predict", 
    json=startup_data
)
prediction = response.json()
print(f"Success probability: {prediction['success_probability']:.3f}")
```

### Prediction with Explanations

```python
import requests

# Get detailed explanation
response = requests.post(
    "http://localhost:8000/predict/explain",
    json=startup_data
)
result = response.json()

print(f"Prediction: {result['prediction']['success_probability']:.3f}")
print("\nTop factors:")
for factor in result['top_factors']:
    print(f"- {factor['feature']}: {factor['importance']:.3f} ({factor['impact']})")
```

### Health Check & System Status

```python
import requests

# Check system health
health = requests.get("http://localhost:8000/health").json()
print(f"System status: {health['status']}")
print(f"Models loaded: {health['models_loaded']}/3")

# Get available options
categories = requests.get("http://localhost:8000/categories").json()
regions = requests.get("http://localhost:8000/regions").json()

print(f"Available categories: {len(categories['categories'])}")
print(f"Available regions: {len(regions['regions'])}")
```

## Model Integration

### Model Loading Strategy

Models are loaded asynchronously during FastAPI startup using the `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models, explainers, and preprocessor
    await load_models()
    yield
    # Shutdown: Cleanup (if needed)
    pass

app = FastAPI(lifespan=lifespan)
```

### Model Selection Logic

- **Default Model**: XGBoost (best F1-score performance: 29.1%)
- **Fallback Strategy**: Uses first available model if XGBoost unavailable
- **Explanation Matching**: SHAP explainers automatically matched to prediction model

### Feature Engineering Pipeline

The preprocessor maintains consistency between training and production:

```python
# Training phase (notebooks)
processor = StartupDataProcessor()
processor.fit(training_data)  # Learn mappings and statistics
processor.save("preprocessor.pkl")

# Production phase (API)
processor = StartupDataProcessor.load("preprocessor.pkl")
features = processor.transform_single(api_request_data)
prediction = model.predict_proba(features.reshape(1, -1))
```

## Production Deployment

### Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY results/ ./results/
COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=../results/models/
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
```

### Performance Optimization

**Startup Performance:**
- Models loaded once during application startup (not per request)
- Preprocessor fitted parameters cached in memory
- Geographic lookup dictionaries precomputed

**Memory Management:**
- Models loaded globally to avoid per request loading overhead
- Feature arrays use NumPy for efficient memory usage
- Garbage collection optimized for ML workloads

### Scaling Considerations

**Horizontal Scaling:**
- Stateless API design enables load balancer distribution
- Models loaded independently by each worker process
- No shared state between requests

**Vertical Scaling:**
- CPU bound workloads benefit from multi core deployment
- Memory requirements: ~500MB per worker for model storage
- Recommended: 2-4 workers per CPU core for optimal throughput

## Error Handling

### Input Validation

```python
class StartupFeatures(BaseModel):
    country_code: str
    region: str
    city: str
    category_list: str
    founded_year: int
    
    @field_validator('founded_year')
    @classmethod
    def validate_founded_year(cls, v):
        if v < 1995 or v > 2015:
            raise ValueError('Founded year must be between 1995 and 2015')
        return v
```

### Error Response Format

```json
{
  "detail": "Prediction failed: Invalid input format",
  "status_code": 422,
  "error_type": "ValidationError"
}
```

### Graceful Degradation

- **Model Unavailable**: Returns HTTP 503 with clear error message
- **Preprocessing Failure**: Provides detailed error context for debugging
- **SHAP Explanation Failure**: Falls back to basic prediction without explanations
- **Geographic Normalization**: Unknown regions/cities default to lowest density tier

## Contributing

This project was developed as a personal learning project. For future questions and/or suggestions:

1. Open an issue describing the enhancement or bug
2. Fork the repository and create a feature branch
3. Follow coding standards
4. Write tests for new functionality
5. Update documentation as needed
6. Submit a pull request with detailed description of changes

## License

This project is open source and available under the MIT License.

## Author

**Ryan Fabrick**
- Statistics and Data Science (B.S) Student, University of California Santa Barbara
- GitHub: [https://github.com/RyanFabrick](https://github.com/RyanFabrick)
- LinkedIn: [www.linkedin.com/in/ryan-fabrick](https://www.linkedin.com/in/ryan-fabrick)
- Email: ryanfabrick@gmail.com

## Acknowledgments & References

- **[Żbikowski, K., & Antosiuk, P. (2021)](https://www.sciencedirect.com/science/article/pii/S0306457321000595)** - "A machine learning, bias-free approach for predicting business success using Crunchbase data." *Information Processing and Management*, 58(4), 102555
- **[Crunchbase](https://www.crunchbase.com/)** - Startup and company database providing the 50,000+ company dataset for model training and validation
- **[XGBoost](https://xgboost.readthedocs.io/)** - Optimized distributed gradient boosting library where machine learning algorithims are implemented under
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library providing preprocessing, modeling, and evaluation tools including logistic regression and SVM implementations
- **[Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)** - Linear classification algorithm using logistic function for binary and multiclass prediction with probabilistic outputs
- **[Support Vector Machine (SVM) with RBF Kernel](https://scikit-learn.org/stable/modules/svm.html#svm-classification)** - Non-linear classification algorithm using radial basis function kernel for complex decision boundaries
- **[SHAP](https://shap.readthedocs.io/)** - (SHapley Additive exPlanations) Model interpretability library enabling prediction explanations
- **[Pandas Community](https://pandas.pydata.org/)** - Data manipulation and analysis library
- **[NumPy Community](https://numpy.org/)** - Fundamental package for scientific computing
- **[Jupyter Project](https://jupyter.org/)** - Interactive computing environment for data analysis, processing, modeling, evaluation, and documentation
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs with Python
- **[Uvicorn](https://www.uvicorn.org/)** - Lightning fast ASGI server for Python web applications
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation library using Python type annotations
- **[React Community](https://react.dev/)** - JavaScript library for building interactive user interfaces
- **[Next.js Community](https://nextjs.org/)** - React framework enabling full stack web applications
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility first CSS framework for rapid UI development

_________________________________________________
Built with ❤️ for the machine learning community
