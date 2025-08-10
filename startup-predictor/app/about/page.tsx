"use client";
import React from 'react';
import { NotebookPen, Code2, GraduationCap, KeyRound, Bot, ChartNetwork, TrendingUp, BookOpen, Database, Brain, Target, Users, Award, GitBranch } from 'lucide-react';
import Link from 'next/link';

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="container-flexible py-3 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">ML Startup Success Predictor</h1>
              <p className="text-gray-600">Machine Learning Powered analysis of startup success potential</p>
            </div>
          </div>
          <Link href="/" 
            className="inline-flex items-center text-blue-600 hover:text-blue-800 font-medium transition-colors">
            ← Back to Predictor
          </Link>
        </div>
      </div>

      {/* Main Content Container - Added to match main page structure */}
      <div className="container-flexible py-8 px-4 sm:px-6 lg:px-8">
        {/* Grid Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* About This Project - Column 1 */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
              <BookOpen className="h-6 w-6 text-blue-600 mr-2" />
              About This Project
            </h2>

            <div className="space-y-3">
              <div className="border-l-4 border-purple-500 pl-4 bg-purple-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Target className="h-4 w-4 text-purple-600 mr-2" />
                  Purpose
                </h3>
                <p className="text-gray-700 leading-relaxed text-sm">
                  <span className="text-black bg-yellow-100 px-1 rounded">Democratize startup success prediction through machine learning</span>. Built on 
                  <span className="text-black bg-blue-100 px-1 rounded">Żbikowski & Antosiuk (2021)</span> methodology using 
                  <span className="text-black bg-green-100 px-1 rounded">50,000+ companies across 25 years</span>.
                  Makes <span className="text-black bg-yellow-100 px-1 rounded">prediction models accessible</span> to help <span className="text-black bg-blue-100 px-1 rounded">validate</span> ideas, identify 
                  <span className="text-black bg-green-100 px-1 rounded">opportunities</span>, and support research with 
                  <span className="text-black bg-purple-100 px-1 rounded">machine learning tools</span> for studying 
                  <span className="text-black bg-orange-100 px-1 rounded">success patterns</span> across industries and cycles.
                </p>
              </div>

              <div className="border-l-4 border-blue-500 pl-4 bg-blue-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Brain className="h-4 w-4 text-blue-600 mr-2" />
                  Machine Learning Approach
                </h3>
                <div className="text-gray-700 leading-relaxed text-sm space-y-3">
                  <p>
                    Tested three algorithms: <span className="text-black bg-yellow-100 px-1 rounded">Logistic Regression with L1</span>, 
                    <span className="text-black bg-blue-100 px-1 rounded">SVM with RBF</span>, and 
                    <span className="text-black bg-green-100 px-1 rounded">XGBoost</span>.
                    Via <span className="text-black bg-purple-100 px-1 rounded">5-fold stratified cross-validation</span>, 
                    <span className="text-black bg-orange-100 px-1 rounded">XGBoost emerged as best</span> and powers the app.
                    Validated against <span className="text-black bg-blue-100 px-1 rounded">Żbikowski & Antosiuk</span> benchmarks, achieving 
                    <span className="text-black bg-green-100 px-1 rounded">67.7% of F1-score target</span>.
                    <span className="text-black bg-purple-100 px-1 rounded">Cross-validation stability</span> showed consistent results. 
                    Integrated <span className="text-black bg-orange-100 px-1 rounded">SHAP explainers</span> provide interpretability of founding-time success factors.
                  </p>
                </div>
              </div>

              <div className="border-l-4 border-red-500 pl-4 bg-green-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <GraduationCap className="h-4 w-4 text-red-600 mr-2" />
                  Academic Paper <span className="font-normal italic ml-2 text-gray-600 text-xs"> (click to access)</span>
                </h3>
                <div className="text-gray-700 leading-relaxed text-sm space-y-3">
                  <p>
                      <a
                      href="https://www.sciencedirect.com/science/article/pii/S0306457321000595"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800"
                      >
                      Żbikowski & Antosiuk (2021)
                      </a>
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Data & Process - Column 2 */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
              <NotebookPen className="h-6 w-6 text-blue-600 mr-2" />
              Data & Process
            </h2>

            <div className="space-y-3">
              <div className="border-l-4 border-green-500 pl-4 bg-green-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Database className="h-4 w-4 text-green-600 mr-2" />
                  Dataset
                </h3>
                <p className="text-gray-700 leading-relaxed text-sm">
                  Crunchbase data of <span className="text-black bg-yellow-100 px-1 rounded">54,000+</span> startups from 
                  <span className="text-black bg-blue-100 px-1 rounded">1902–2014</span>, covering 
                  <span className="text-black bg-purple-100 px-1 rounded">over a century</span> of funding, location, and industry info across 
                  <span className="text-black bg-orange-100 px-1 rounded">750+ global sectors</span>.
                </p>
              </div>

              <div className="border-l-4 border-yellow-500 pl-4 bg-yellow-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <GitBranch className="h-4 w-4 text-yellow-600 mr-2" />
                  Documented Notebooks <span className="font-normal italic ml-2 text-gray-600 text-xs"> (click to view)</span>
                </h3>
                    <ul className="text-gray-700 leading-relaxed space-y-2">
                    <li className="flex items-center bg-white p-2 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                        <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mr-3 flex-shrink-0"></span>
                        <a
                        href="https://github.com/RyanFabrick/ML-Startup-Success-Prediction/blob/2d72964bd8447d5ffbff341e6a172381fd62bd8f/notebooks/01_data_exploration.ipynb"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 font-medium text-xs"
                        >
                        Notebook 1: Exploratory Data Analysis
                        </a>
                    </li>
                    <li className="flex items-center bg-white p-2 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                        <span className="inline-block w-2 h-2 bg-purple-500 rounded-full mr-3 flex-shrink-0"></span>
                        <a
                        href="https://github.com/RyanFabrick/ML-Startup-Success-Prediction/blob/2d72964bd8447d5ffbff341e6a172381fd62bd8f/notebooks/02_data_preprocessing_feature_engineering.ipynb"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 font-medium text-xs"
                        >
                        Notebook 2: Data Preprocessing & Feature Engineering
                        </a>
                    </li>
                    <li className="flex items-center bg-white p-2 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                        <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-3 flex-shrink-0"></span>
                        <a
                        href="https://github.com/RyanFabrick/ML-Startup-Success-Prediction/blob/2d72964bd8447d5ffbff341e6a172381fd62bd8f/notebooks/03_modeling.ipynb"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 font-medium text-xs"
                        >
                        Notebook 3: Model Development
                        </a>
                    </li>
                    <li className="flex items-center bg-white p-2 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                        <span className="inline-block w-2 h-2 bg-red-500 rounded-full mr-3 flex-shrink-0"></span>
                        <a
                        href="https://github.com/RyanFabrick/ML-Startup-Success-Prediction/blob/2d72964bd8447d5ffbff341e6a172381fd62bd8f/notebooks/04_evaluation.ipynb"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 font-medium text-xs"
                        >
                        Notebook 4: Model Evaluation
                        </a>
                    </li>
                    <li className="flex items-center bg-white p-2 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                        <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full mr-3 flex-shrink-0"></span>
                        <a
                        href="https://github.com/RyanFabrick/ML-Startup-Success-Prediction/blob/2d72964bd8447d5ffbff341e6a172381fd62bd8f/notebooks/05_pipeline_setup.ipynb"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 font-medium text-xs"
                        >
                        Notebook 5: Pipeline Setup
                        </a>
                    </li>
                    </ul>
              </div>

              {/* Technology Stack - Combined with color legend */}
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Code2 className="h-5 w-5 text-rose-600 mr-2" />
                  Technology Stack
                </h3>
                
                {/* Color Legend */}
                <div className="flex items-center justify-center gap-4 mb-3 text-xs text-gray-600">
                  <div className="flex items-center">
                    <span className="w-2 h-2 bg-blue-500 rounded-full mr-1"></span>
                    Full Stack
                  </div>
                  <div className="flex items-center">
                    <span className="w-2 h-2 bg-purple-500 rounded-full mr-1"></span>
                    Machine Learning & Data Science
                  </div>
                </div>

                {/* All Technologies Combined */}
                <div className="flex flex-wrap gap-1.5">
                  {/* Full Stack Technologies */}
                  {['React', 'Next.js', 'Tailwind CSS', 'TypeScript', 'FastAPI', 'Python', 'Pydantic', 'Uvicorn'].map((tech) => (
                    <span key={tech} className="px-1.5 py-0.5 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {tech}
                    </span>
                  ))}
                  {/* ML & Data Science Technologies */}
                  {['XGBoost', 'Logistic Regression', 'SVM', 'SHAP', 'Scikit-learn', 'pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn', 'Plotly.js'].map((tech) => (
                    <span key={tech} className="px-1.5 py-0.5 bg-purple-100 text-purple-800 text-xs rounded-full">
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Performance & Usage - Column 3 */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
                <Award className="h-6 w-6 text-blue-600 mr-2" />
                Performance & Usage
            </h2>

            <div className="space-y-3">
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <ChartNetwork className="h-4 w-4 text-purple-600 mr-2" />
                  Model Performance <span className="font-normal italic ml-2 text-gray-600 text-xs"> (XGBoost)</span>
                </h3>
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
                  <div className="grid grid-cols-2 gap-2 text-center">
                    {/* F1-Score */}
                    <div className="bg-white p-2 rounded-lg shadow-sm">
                      <div className="text-2xl font-bold text-blue-600">29.1%</div>
                      <div className="text-sm text-gray-600">F1-Score</div>
                    </div>
                    
                    {/* AUC-ROC Score */}
                    <div className="bg-white p-2 rounded-lg shadow-sm">
                      <div className="text-2xl font-bold text-purple-600">79.0%</div>
                      <div className="text-sm text-gray-600">AUC-ROC Score</div>
                    </div>
                    
                    {/* Recall */}
                    <div className="bg-white p-2 rounded-lg shadow-sm">
                      <div className="text-2xl font-bold text-green-600">38.8%</div>
                      <div className="text-sm text-gray-600">Recall</div>
                    </div>
                    
                    {/* Precision */}
                    <div className="bg-white p-2 rounded-lg shadow-sm">
                      <div className="text-2xl font-bold text-orange-600">23.4%</div>
                      <div className="text-sm text-gray-600">Precision</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-l-4 border-orange-500 pl-4 bg-orange-50/30 py-3 rounded-r-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Users className="h-4 w-4 text-orange-600 mr-2" />
                  Who Can Use This
                </h3>
                <p className="text-gray-700 leading-relaxed text-sm">
                    <span className="text-black bg-yellow-100 px-1 rounded">ENTREPRENEURS</span> validate & benchmark 
                    ideas, <span className="text-black bg-blue-100 px-1 rounded">INVESTORS</span> identify high potential opportunities, 
                    <span className="text-black bg-green-100 px-1 rounded">RESEARCHERS</span> explore unbiased prediction 
                    methods & trends, and <span className="text-black bg-purple-100 px-1 rounded">STUDENTS</span> learn data driven business analysis via machine learning.
                </p>
              </div>

              {/* Key Features - Moved from column 2 */}
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                    <KeyRound className="h-5 w-5 text-yellow-600 mr-2" />
                    Key Features
                </h3>
                <div className="grid grid-cols-3 gap-3">
                    <div className="bg-gray-50 rounded-lg p-3 flex flex-col items-center text-center">
                    <div className="text-black font-bold text-base mb-1">+3</div>
                    <div className="font-medium text-gray-800 text-xs">Geographic Factors</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 flex flex-col items-center text-center">
                    <div className="text-black font-bold text-base mb-1">+15</div>
                    <div className="font-medium text-gray-800 text-xs">Industry Categories</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 flex flex-col items-center text-center">
                    <div className="text-black font-bold text-base mb-1">+4</div>
                    <div className="font-medium text-gray-800 text-xs">Temporal Patterns</div>
                    </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm mt-6">
          <p>Powered by an XGBoost ML model trained on data with over 50,000 startups</p>
          <p className="mt-1">
            <a 
              href="https://github.com/RyanFabrick/Startup-Success-Prediction.git" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              Click here to access GitHub Repository for this project
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;