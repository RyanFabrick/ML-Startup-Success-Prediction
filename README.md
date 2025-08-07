# *Machine Learning Startup Success Predictor*

A full-stack machine learning application that predicts startup success using over 50,000+ company data points spanning 1990-2015. Built with peer reviewed academic validation methodology and powered by XGBoost. Prior to full-stack implementation, comprehensive analysis was conducted through five documented notebooks: exploratory data analysis, preprocessing and feature engineering, modeling development, performance evaluation, and production pipeline setup.

## Table of Contents
- [Frontend, Backend, Data, & Notebook READMEs (More Detail & Visual Examples)](#frontend-backend-data--notebook-readmes-more-detail--visual-examples)
- [Overview](#overview)
- [Why Did I Build This?](#why-did-i-build-this)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Demo GIFs](#demo-gifs)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [Methodology & Academic Foundation](#methodology--academic-foundation)
- [Overall Model Performances](#overall-model-performances)
- [Use Cases](#use-cases)
- [API Documentation](#api-documentation)
- [Academic Context](#academic-context)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)


## Frontend, Backend, Data, & Notebook READMEs (More Detail & Visual Examples)

For more **comprehensive**, **specific**, and **thorough** documentation and examples:
- [Frontend README](startup-predictor/README.md)
- [Backend README](src/README.md)
- [Data README](data/README.md)
- [Notebooks README](notebooks/README.md)

## Overview

This project implements and extends the bias free startup success prediction methodology from Żbikowski & Antosiuk (2021). This repository provides:

- **Machine Learning Models**: XGBoost, Logistic Regression, and SVM with documentation, analysis, and evaluation
- **Interactive Web Application**: React/Next.js frontend with FastAPI backend
- **Model Interpretability**: SHAP explanations for individual predictions
- **Academic Validation**: Reproduces and extends published research methodology

### Key Results (XGBoost Model)
- **F1-Score**: 29.1% 
- **AUC-ROC**: 79.0% 
- **Recall**: 38.8% 
- **Precision**: 23.4%

## Why Did I Build This?

FIll in later

## Key Features

### Machine Learning Pipeline
- **22 Engineered Features** across geographic, industry, and temporal dimensions
- **Bias Prevention** using only founding-time information
- **Cross-Validation** with 5-fold stratified approach
- **SHAP Integration** for model interpretability

### Web Application
- **Real-time Predictions** with confidence intervals
- **Interactive UI** with searchable dropdowns for 750+ regions/cities
- **Multi-category Selection** from 15 industry categories
- **Visual Explanations** showing key success factors

## System Architecture

```
Fill in later
```

## Demo GIFs

Fill in later

## Technology Stack

### Machine Learning & Data Science
- **Python**:
- **XGBoost**:
- **Logisti Regression with Regularization**:
- **SVM with RBF Kernel**:
- **SHAP**: 
- **Jupyter**: 
- **Pandas**:
- **NumPy**:
- **Matplotlib**:
- **Seaborn**:
- **scikit-learn**:

### Frontend
- **React**:
- **Next.js**:
- **Typescript**:
- **Tailwind CSS**:

### Backend
- **FastAPI**:
- **Pydantic**:
- **Uvicorn**:
- **Data Processing**:


## Project Structure
```
Fill in later
```

## Quick Start

### Prerequisites
- Python 3.8+S
- Node.js 16+
- pip and npm

### 1. Clone Repository
```bash
git clone https://github.com/RyanFabrick/Startup-Success-Prediction.git
cd Startup-Success-Prediction
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
cd app
python app.py
# Server runs on http://localhost:8000
```

### 3. Frontend Setup
```bash
# Install Node dependencies
cd startup-predictor
npm install

# Start development server
npm run dev
# Application runs on http://localhost:3000
```

### 4. API Health Check
```bash
curl http://localhost:8000/health
```

### Environment Variables

The application requires environment variables to be configured for proper operation.

```bash
# Environment (.env)
cp .env.example .env
# Configure settings as needed
```

## Notebooks

The complete data process and analysis is documented across five notebooks:

1. **[01_EDA](notebooks/01_data_exploration.ipynb)** 
2. **[02_Preprocessing_&_Feature_Engineering](notebooks/02_data_preprocessing_feature_engineering.ipynb)** 
3. **[03_Modeling](notebooks/03_modeling.ipynb)** 
4. **[04_Evaluation](notebooks/04_evaluation.ipynb)** 
5. **[05_Pipeline_Setup](notebooks/05_pipeline_setup.ipynb)**

Each notebook is self contained with thorughly detailed documentation for each step and can be run independently. Go to the [Notebooks README](notebooks/README.md) for more information.

## Methodology & Academic Foundation

### Research Validation
Based on **"A machine learning, bias-free approach for predicting business success using Crunchbase data"** (Żbikowski & Antosiuk, 2021). In my implementation I attempt to:

- **Reproduces** the original bias-free methodology
- **Extends** with enhanced feature engineering (22 vs 8 features)
- **Validates** across multiple economic cycles (1995-2015)
- **Compares** academic vs practical success definitions

### Feature Engineering
- **Geographic Factors** (3): Region/city startup density rankings, US indicator
- **Industry Categories** (15): Binary encoding for major startup sectors
- **Temporal Features** (4): Standardized founding year, economic era classification

### Success Definition
**Academic Success**: Company acquired OR (still operating AND reached Series B funding)
- Eliminates look ahead bias by using only founding time features
- Focuses on observable outcomes rather than subjective metrics

## Overall Model Performances

```
| Model | Precision | Recall | F1-Score | AUC-ROC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.169 | 0.709 | 0.273 | 0.781 |
| SVM (RBF) | 0.155 | 0.689 | 0.252 | 0.740 |
| XGBoost | 0.234 | 0.388 | 0.291 | 0.790 |
| Academic Target | 0.570 | 0.340 | 0.430 | NaN |
```

## Use Cases

### For Entrepreneurs
- **Validate business ideas** against historical success patterns
- **Identify key risk factors** before launching
- **Benchmark** against similar companies

### For Investors
- **Screen opportunities** with data driven insights
- **Supplement due diligence** with quantitative analysis
- **Understand** geographic and industry trends

### For Students and Researchers
- **Academic validation** of published methodologies
- **Study** startup ecosystem patterns
- **Explore** bias-free prediction techniques

## API Documentation

### Core Endpoints
- `POST /predict` - Basic success prediction
- `POST /predict/explain` - Prediction with SHAP explanations
- `GET /categories` - Available industry categories
- `GET /regions` - Searchable region list
- `GET /cities` - Searchable city list
- `GET /health` - System status

### Example Request
```python
import requests

data = {
    "country_code": "USA",
    "region": "SF Bay Area",
    "city": "San Francisco",
    "category_list": "software mobile",
    "founded_year": 2010
}

response = requests.post("http://localhost:8000/predict/explain", json=data)
prediction = response.json()
```

## Academic Context

### Literature Foundation
This project validates and extends the methodology from:

Żbikowski, K., & Antosiuk, P. (2021). A machine learning, bias-free approach for predicting business success using Crunchbase data. *Information Processing and Management*, 58(4), 102555.

This study presents an academically and technically comprehensive machine learning approach to predict startup success while explicitly addressing the **look-ahead bias** problem that plagues most existing research in this domain. The authors analyzed 213,171 companies from the Crunchbase database to develop practically applicable prediction models. While numerous studies have attempted to predict business success using machine learning, they typically suffer from methodological flaws that make their results impractical for actual investment decisions.

This research establishes a new standard for startup success prediction by prioritizing practical applicability over theoretical performance, providing a valuable tool for data-driven investment decisions while advancing our understanding of entrepreneurial success factors. I used it as both context and inspiration for this project!

### Key Contributions
1. **Independent validation** using separate dataset
2. **Enhanced feature engineering** with funding progression metrics
3. **Temporal robustness** across multiple economic cycles
4. **Production deployment** with interactive explanations

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

- **[CDIP (Coastal Data Information Program)](https://cdip.ucsd.edu/)** - Buoy oceanographic data source
- **[The Surfers View](https://www.thesurfersview.com/)** - Live surf camera feed provider
- **[Roboflow](https://roboflow.com/)** - Computer vision and machine learning model training infrastructure
- **[FFmpeg](https://ffmpeg.org/)** - Professional video stream processing and frame extraction capabilities
- **[Flask Community](https://flask.palletsprojects.com/)** - Excellent web framework
- **[React Community](https://react.dev/)** - Super helpful and clear documentation

________________________________________
Built with ❤️ for the ....

Fill in later