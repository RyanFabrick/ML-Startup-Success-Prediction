# *Data*

## Table of Contents
- [Overview](#overview)
- [Folder Structure & Content](#folder-structure--content)
- [Data Source](#data-source)
- [Purpose](#purpose)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)

## Overview

This folder contains all datasets used in the Startup Success Predictor project, organized into raw source data and processed feature sets ready for the machine learning models and pipeline application setup.

## Folder Structure & Content

### `/raw`
- **`startups_data.csv`** - Original Crunchbase dataset with 50,000+ startup records (1990-2015)
- **`startups_data.xlsx`** - Excel version of the raw dataset

### `/processed`
- **`startup_data_processed.csv`** - Clean dataset after preprocessing pipeline
- **`unique_cities.csv`** - Reference list of 750+ cities for frontend dropdowns
- **`unique_regions.csv`** - Reference list of regions for user selection
- **`X_train_scaled.csv`** - Scaled training features (22 dimensions)
- **`X_train_stratified.csv`** - Stratified training set for cross-validation
- **`X_train_temporal.csv`** - Temporally split training data
- **`X_test_scaled.csv`** - Scaled test features
- **`X_test_stratified.csv`** - Stratified test set
- **`X_test_temporal.csv`** - Temporal test set
- **`y_train_stratified.csv`** - Training labels (stratified)
- **`y_train_temporal.csv`** - Training labels (temporal split)
- **`y_test_stratified.csv`** - Test labels (stratified)
- **`y_test_temporal.csv`** - Test labels (temporal split)

## Data Source

The raw dataset originates from Crunchbase, accessible on [Kaggle](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase) a comprehensive database of startup companies, and follows the methodology established in [Żbikowski & Antosiuk (2021)](https://www.sciencedirect.com/science/article/pii/S0306457321000595) for bias free startup success prediction

## Purpose

- **Raw data**: Source material for exploratory analysis and feature engineering
- **Processed data**: Ready to use datasets for model training, validation, and production deployment
- **Reference files**: Support interactive web application with searchable location options

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