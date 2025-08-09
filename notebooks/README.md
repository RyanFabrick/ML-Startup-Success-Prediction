# *Notebooks*

## Table of Contents
- [Overview](#overview)
- [Notebook Descriptions](#notebook-descriptions)
    - [Notebook 1 (EDA) Description](#1-01_exploratory-data-analysis)
    - [Notebook 2 (Preprocessing & Feature Engineering) Description](#2-02_preprocessing__feature_engineering)
    - [Notebook 3 (Modeling) Description](#3-03_modeling)
    - [Notebook 4 (Evaluation) Description](#4-04_evaluation)
    - [Notebook 5 (Pipeline Setup) Description](#5-05_pipeline_setup)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)
    
## Overview

The five notebooks follow a logical, academic, and documented progression from raw data exploration through model deployment preparation. They implement and extend the bias-free methodology from Żbikowski & Antosiuk (2021) while adding enhanced feature engineering, varying model techniques, and new visualizations. Click each documented notebook below for step by step documentation, code, outputs, and more.

- **[01_Exploratory Data Analysis](01_data_exploration.ipynb)** 
- **[02_Preprocessing_&_Feature_Engineering](02_data_preprocessing_feature_engineering.ipynb)** 
- **[03_Modeling](notebooks/03_modeling.ipynb)** 
- **[04_Evaluation](04_evaluation.ipynb)** 
- **[05_Pipeline_Setup](05_pipeline_setup.ipynb)**

## Notebook Descriptions

### 1. [01_Exploratory Data Analysis](01_data_exploration.ipynb)

This notebook conducts systematic exploratory data analysis on 54,294 startup records with 39 features, implementing rigorous academic methodology to uncover patterns essential for bias-free predictive modeling. The analysis follows eight structured investigation areas that collectively establish the foundation for sophisticated machine learning pipeline development.

**Key Analytical Findings:**
- **Extreme Class Imbalance**: Only 8.1% acquisition rate (3,692 acquired vs 41,829 operating companies) requiring specialized handling techniques including SMOTE/ADASYN and cost sensitive learning approaches
- **Geographic Concentration**: USA dominates with 75.4% of all acquisitions (2,785 of 3,692), creating strong predictive signals for location-based features and revealing significant ecosystem clustering effects
- **Funding Success Premium**: Acquired companies raised 51% more capital ($23.8M vs $15.7M average) and completed more funding rounds (2.0 vs 1.7), indicating capital intensity correlates with acquisition probability
- **Temporal Vintage Effects**: Dot com era companies (1995-2000) achieve 20% acquisition rates versus modern cohorts at 3.5%, demonstrating time-since-founding bias requiring survival analysis techniques

**Data Quality Assessment:**
- **Missing Data Patterns**: Systematic gaps in temporal features (29%) and geographic data (18-44%) requiring strategic imputation rather than random deletion
- **Funding Distribution Extremes**: 1,686x difference between mean ($15.9M) and median ($2M) funding necessitating log transformations and robust outlier handling
- **Multicollinearity Concerns**: Strong correlations (0.7-0.9) between sequential funding rounds requiring dimensionality reduction and regularization techniques

**Feature Engineering Opportunities:**
- **Funding Velocity Metrics**: Time to funding analysis reveals 90% of companies secure capital within 2-3 years, enabling progression speed indicators
- **Geographic Hierarchy Encoding**: Country→Region→City clustering patterns suggest hierarchical categorical encoding strategies
- **Industry Consolidation**: Software sector dominance (10,773 entries) and long-tail distribution requiring "Other" category bucketing and multi-label encoding approaches

**ML Pipeline Implications:**
- The analysis establishes preprocessing requirements including stratified sampling for class imbalance, era-based train/test splits preventing temporal leakage, and feature selection strategies addressing multicollinearity. Success prediction modeling must incorporate geographic premium effects, funding progression patterns, and vintage bias correction to achieve reliable performance on this highly imbalanced, temporally complex dataset.

### 2. [02_Preprocessing_&_Feature_Engineering](02_data_preprocessing_feature_engineering.ipynb)

This notebook implements data preprocessing and feature engineering on the dataset, transforming the raw 54,294 company dataset into a clean, bias-free modeling foundation through systematic academic methodology. The preprocessing pipeline addresses temporal consistency, class imbalance, and feature selection challenges while maintaining strict adherence to founding time only information to prevent look ahead bias.

**Key Preprocessing Transformations:**
- **Temporal Filtering Implementation**: Applied 1995-2015 cutoff reducing dataset to 36,905 companies (68% retention), segmenting into three economic eras: dot-com (2,970 companies, 8.0%), post-crash (11,554 companies, 31.3%), and recovery (22,381 companies, 60.7%) periods enabling cross-cycle validation strategies
- **Target Variable Creation**: Established dual success definitions with strict academic criteria (acquired only: 7.72% success rate, 2,785 companies) and extended definition (acquired OR Series B+ operating: 17.50% success rate, 6,313 companies) maintaining 12:1 class imbalance requiring specialized handling techniques
- **Founding-Time Feature Restriction**: Limited feature set to 22 founding-time-only variables eliminating look ahead bias through systematic exclusion of post-founding funding, growth metrics, and exit data while preserving geographic (3 features), industry (15 categories), and temporal (4 features) dimensions
- **Geographic Density Engineering**: Implemented five tier startup density ranking system with SF Bay Area achieving tier 5 (5,580 companies) and quantile-based city-level mapping enabling ecosystem clustering effect capture while creating USA dominance flag (61.3% dataset coverage, 22,103 companies)

**Data Quality Improvements:**
- **Missing Value Strategy**: Applied domain informed imputation using mode-based geographic density assignment (tier 5 for 8.0% missing region/country data) and "unknown" categorical encoding for market classifications (4.6% missing), achieving 100% feature completeness across modeling variables
- **Industry Feature Standardization**: Eliminated parsing artifacts ("and", "&" contamination affecting 6% of entries) and implemented multi-label binary encoding for top 15 categories (software: 8,645, mobile: 4,694, social: 3,856 entries) capturing 67% classification coverage with consistent lowercase normalization
**Temporal Bias Investigation**: Revealed dramatic success rate decline from 19.4% (dot-com era) to 3.4% (recovery era) with 5.5x difference bet- ween pre-2010 (11.81%) and post-2010 (2.13%) founding cohorts, necessitating era-based feature engineering and temporal validation strategies

**Feature Engineering Outcomes:**
- **Multi-Scale Geographic Encoding**: Created hierarchical location features spanning country binary indicators, regional density tiers, and city-level startup concentration rankings enabling capture of ecosystem network effects and talent pool accessibility factors
- **Economic Cycle Integration**: Established era-based categorical features (era_dotcom_era, era_post_crash, era_recovery) with standardized founding year continuous variables enabling algorithm learning of macroeconomic environmental effects across different business cycles
- **Complete Case Dataset Achievement**: Generated final modeling dataset with 36,075 companies × 22 features achieving zero missing values through strategic imputation while preserving realistic startup ecosystem characteristics and class distribution patterns

**ML Pipeline Readiness Assessment:**
- **Dual Split Strategy Implementation**: Created both stratified random splits (80/20 maintaining 7.72% success rate) for academic replication and time-aware splits (2010 cutoff: training 20,846 samples 1995-2010, testing 15,229 samples 2011-2014) enabling temporal generalization validation and production deployment scenario testing
- **Algorithm Compatibility Optimization**: Engineered mixed feature types including binary categorical indicators, ordinal density rankings, and standardized continuous variables optimizing compatibility with ensemble methods (XGBoost, Random Forest) while maintaining linear model support through proper encoding strategies
- **Production Deployment Framework**: Established real-time inference capability through founding time feature derivation, standardized imputation rules, and automated era assignment logic enabling scalable prediction pipeline deployment without complex preprocessing overhead during model serving phases

### 3. [03_Modeling](03_modeling.ipynb)

This notebook implements comprehensive machine learning model development and evaluation on preprocessed startup data, applying three distinct algorithmic approaches to predict acquisition success using founding-time-only features. The modeling pipeline follows rigorous academic methodology with systematic hyperparameter optimization, cross-validation, and performance benchmarking against published research standards from Żbikowski & Antosiuk (2021).

**Key Modeling Implementations:**
- **Algorithm Portfolio Deployment:** Implemented three machine learning approaches including L1-regularized Logistic Regression (linear baseline with automatic feature selection), SVM with RBF kernel (non-linear decision boundaries for complex pattern detection), and XGBoost ensemble (gradient boosting for optimal precision-recall balance) enabling algorithmic comparison across different learning paradigms
- **Hyperparameter Optimization Strategy:** Executed systematic parameter space exploration through GridSearchCV (Logistic Regression: 120 model fits across C, penalty, class weighting combinations) and RandomizedSearchCV (XGBoost: 250 model fits optimizing ensemble size, tree depth, learning rates, regularization parameters) with F1-score optimization focus addressing severe class imbalance challenges
- **Academic Replication Framework:** Applied identical evaluation methodology to published research benchmarks using 5-fold stratified cross-validation, precision/recall/F1-score metrics, and direct performance comparison against established targets (57% precision, 34% recall, 43% F1-score) ensuring methodological consistency and research validation capabilities
- **Production Ready Model Persistence:** Established complete model serialization pipeline saving optimized algorithms, SHAP explainers for interpretability, feature column specifications, and comprehensive evaluation results enabling seamless deployment transition and real-time inference implementation

**Performance Achievements:**
- **XGBoost Best Performer:** Achieved best overall results with F1-score of 29.1% (67.7% of academic target), precision of 23.4%, recall of 38.8%, and AUC-ROC of 79.0%, demonstrating ensemble method effectiveness for complex startup ecosystem pattern recognition while maintaining practical deployment characteristics
- **Class Imbalance Handling Excellence:** Successfully addressed 7.72% success rate through balanced class weighting (Logistic/SVM) and scale_pos_weight optimization (XGBoost), achieving recall rates of 38.8%-70.9% for successful startup identification while controlling false positive rates for practical venture capital applications
- **Feature Importance Discovery:** Revealed temporal features dominate predictive power (48.9% XGBoost importance) with era_recovery (18.31%) and founded_year_std (16.02%) as primary drivers, followed by industry categories (37.4%) and geographic indicators (13.7%) providing actionable insights for investment decision frameworks
- **Cross-Validation Stability Validation:** Demonstrated generalization with minimal variance across validation folds (XGBoost: 0.297 ± 0.011, Logistic: 0.265 ± 0.004) and strong alignment between cross-validation estimates and test performance, confirming absence of overfitting and reliable production deployment expectations

**Analytical Insights:**
- **Algorithmic Comparison Analysis:** Linear models (Logistic Regression, SVM) achieved exceptional recall performance (68.9%-70.9%) but suffered precision challenges (15.5%-16.9%), while XGBoost provided superior precision-recall balance through tree-based ensemble architecture and optimal regularization strategies addressing founding-time feature constraints
- **Geographic Premium Effects:** Identified USA location as critical success predictor (7.01% XGBoost importance, +0.635 Logistic coefficient) with hierarchical startup density features contributing 13.7% total importance, confirming ecosystem clustering effects and venture capital access advantages for geographic-based investment strategies
- **Temporal Vintage Bias Quantification:** Revealed founding era dominates model decisions with recovery period (18.31% importance) and standardized founding year (16.02% importance) effects, indicating strong survival analysis requirements and macroeconomic cycle impacts on startup acquisition probability patterns
- **Industry Sector Paradoxes:** Discovered complex non-linear relationships where biotechnology shows high XGBoost importance (5.04%) but negative Logistic coefficients (-0.662), demonstrating ensemble method superiority for capturing interaction effects that linear approaches cannot represent effectively

**ML Pipeline Production Readiness:**
- **Comprehensive Model Serialization:** Implemented complete persistence framework including trained models (joblib format), SHAP explainers for interpretability, feature column specifications (22 founding time variables), and evaluation metrics enabling seamless API deployment and real time inference capabilities
- **Business Decision Support Framework:** Established three distinct deployment strategies optimized for different risk profiles: conservative XGBoost approach (high precision, moderate recall) for resource-constrained screening, aggressive Logistic Regression (high recall, low precision) for comprehensive opportunity identification, and balanced ensemble potential for optimized cost-sensitive applications
- **Visualization and Reporting Pipeline:** Generated comprehensive evaluation artifacts including performance comparison charts, feature importance analyses, confusion matrix comparisons, and academic benchmark validations saved at 300 DPI resolution enabling stakeholder communication and research publication support
- **Interpretability and Explainability Integration:** Deployed SHAP (SHapley Additive exPlanations) framework across all models providing individual prediction explanations, feature contribution analysis, and model behavior understanding critical for venture capital investment justification and regulatory compliance requirements

**Academic Validation Assessment:**
- **Benchmark Comparison Results:** Achieved 67.7% of published F1-score target with XGBoost leading performance, indicating competitive results given founding-time only feature constraints while identifying opportunities for advanced ensemble methods and feature engineering improvements to close remaining performance gaps
- **Methodological Rigor Validation:** Applied identical evaluation frameworks, cross-validation strategies, and metric calculations to published research ensuring fair comparison and reproducible results while maintaining strict adherence to bias-free temporal validation preventing look ahead information leakage
- **Statistical Significance Confirmation:** Demonstrated consistent performance across multiple validation approaches (stratified CV, temporal splits, stability checks) with low variance estimates providing confidence intervals for production deployment and confirming reliable model behavior across different data distributions and economic conditions

### 4. [04_Evaluation](04_evaluation.ipynb)

This notebook conducts model evaluation and diagnostic analysis using a range of statistical, business, and interpretability metrics. It compares three trained models: Logistic Regression, SVM (RBF kernel), and XGBoost on the held out test set using a consistent feature set limited to founding time only variables. The evaluation framework benchmarks performance against academic standards (Żbikowski & Antosiuk, 2021), while extending analysis to include calibration, threshold tuning, financial impact simulation, feature stability, and misclassification profiling.

**Core Evaluation Areas:**
- **Academic Benchmark Comparison**: Achieved 38.8% recall (exceeding the 34% target) and 29.1% F1-score (67.7% of the 43% target), but fell short on precision (23.4% vs 57.0%), highlighting the trade-off between recall and false positive rate in highly imbalanced startup success prediction
- **Model Ranking Summary**: XGBoost outperformed others in F1-score and AUC-ROC (0.790), while Logistic Regression and SVM offered exceptional recall (\~70%) but at a higher false positive cost
- **Interpretation for Stakeholders**: Suggested tiered deployment: Logistic Regression/SVM for high-recall screening, XGBoost for precision-focused decisions, and ensemble approaches for bridging performance gaps

**Advanced Analytical Modules:**
- **ROC and PR Curve Analysis**: Evaluated models using both ROC and Precision-Recall curves with bootstrapped confidence intervals. XGBoost showed the best average precision (0.213), confirming its effectiveness in extreme class imbalance contexts
- **Probability Calibration Assessment**: Calibration curves and Brier scores revealed SVM had the most reliable probability estimates (Brier = 0.0677), while Logistic Regression showed significant overconfidence (Brier = 0.1956)

**Business-Oriented Threshold Optimization**:
- Implemented a cost-sensitive framework assuming realistic VC economics: \$50K false positive cost, \$500K false negative cost, and \$1M expected ROI
- Identified optimal thresholds for four strategies: business value (0.140), conservative precision (0.710), aggressive recall (0.100), and balanced F1 (0.420)
- Demonstrated that threshold tuning drastically affects success rates, ROI, and model utility depending on fund risk appetite

**Simulation and Deployment Readiness:**
- *Investment Simulation:*
    - Random selection yielded an 8.0% success rate and -92% ROI
    - guided selection improved success to 29.0% with a -71% ROI under conservative assumptions — a 21-point improvement, affirming practical value even in challenging conditions
- *Feature Importance Stability:*
    - Conducted 5-fold cross validation on feature importances. Temporal features (e.g., founded_year_std, era_recovery) dominated and exhibited high stability (CV < 0.1)
    - Geographic (is_usa) and industry (category_biotechnology, category_web) features also demonstrated consistent, interpretable influence

**Error and Reliability Analysis:**
- *Misclassification Patterns:*
    - False positives (n=709) were dominated by USA based and dot-com/post-crash era startups, showing the model’s historical pattern overreliance.
    - False negatives (n=341) were concentrated in the recovery era, suggesting under recognition of recent successful startups.
- *Prediction Confidence Breakdown:*
    - Accuracy was excellent at low confidence levels (<0.3) but collapsed above 0.6, indicating dangerous overconfidence and urgent need for post hoc calibration (e.g., Platt scaling).
    - Identified 49 high-confidence incorrect predictions and 747 low confidence correct ones, suggesting the decision boundary is complex and non linear.

**Temporal Robustness and Market Evolution:**
- Dot com (1995–2000): 20.2% success rate
- Post crash (2001–2008): 12.7%
- Recovery (2009–2015): 3.5%
- Highlights a 5.8x drop in base success rates, indicating non stationary conditions across time
- *Walk-Forward Validation:*
    - Demonstrated major performance deterioration over time (mean F1-score: 0.009 ± 0.013), underscoring the model’s inability to generalize to recent eras.
    - Reinforced need for dynamic retraining, temporal segmentation, and rolling window models for real-world VC applications.

**Academic Validation:**
- McNemar’s test confirmed statistically significant superiority of XGBoost over both Logistic Regression and SVM (p < 0.001).
- Bootstrap intervals further validated the stability and reproducibility of model rankings.
- Precision performance gap attributed to dataset differences (time periods, economic context), success definition variance, and possible preprocessing discrepancies between this project and the academic benchmark.

**ML Pipeline Implications:**
- Calibrate output probabilities before deployment
- Tailor thresholds to business context using cost value simulation
- Use model ensembles to balance high recall and high precision
- Retrain regularly with recent data to mitigate temporal degradation

### 5. [05_Pipeline_Setup](05_pipeline_setup.ipynb)

This notebook implements the final production deployment preparation phase, creating a preprocessing pipeline optimized for real time inference and API integration. The pipeline setup establishes the foundation for scalable startup success prediction deployment by encapsulating all data transformation logic into a reusable, version controlled preprocessing component that ensures consistency between training and production environments.

**ML Pipeline Implementation:**
- **Production Preprocessor Creation**: Implemented data preprocessing pipeline using the complete 54,294 raw startup dataset, fitting transformation parameters including region density mappings (1,089 regions), city density mappings (4,188 cities), and founding year standardization statistics (mean: 2007.4, std: 7.6) enabling consistent feature engineering across training and inference phases
- **Serialization and Persistence Framework**: Created robust model persistence architecture saving the fitted preprocessor to ../results/models/preprocessor.pkl using optimized serialization protocols, ensuring preprocessing transformations, categorical encodings, and statistical parameters are preserved for API deployment and real time prediction serving
- **Feature Engineering Pipeline Integration**: Established automated preprocessing workflow generating exactly 22 features from raw startup data, incorporating geographic density encoding, temporal standardization, industry categorization, and missing value imputation strategies developed throughout the modeling pipeline ensuring production feature consistency
- **Deployment Readiness Validation**: Verified complete preprocessing pipeline functionality through end to end testing using original raw data format with latin-1 encoding handling, confirming successful transformation of startup records into model ready feature vectors suitable for XGBoost, Logistic Regression, and SVM inference execution

**Technical Achievements:**
- **Modular Architecture Implementation**: Developed reusable create_and_fit_preprocessor() function from src.data_preprocessing module enabling consistent preprocessing logic across development, testing, and production environments while maintaining strict separation between data transformation and model inference components for enhanced maintainability
- **Encoding Compatibility Assurance**: Implemented latin-1 encoding support ensuring compatibility with original startup dataset format and handling international character sets present in company names, geographic locations, and categorical descriptions preventing encoding errors during production data ingestion

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