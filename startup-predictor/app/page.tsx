"use client";
import React, { useState, useEffect } from 'react';
import { AlertCircle, TrendingUp, BarChart3, Lightbulb, Loader2 } from 'lucide-react';

// Types
interface PredictionResponse {
  success_probability: number;
  prediction: number;
  model_used: string;
  confidence: string;
}

interface ExplanationResponse {
  prediction: PredictionResponse;
  feature_importance: Record<string, number>;
  top_factors: Array<{
    feature: string;
    importance: number;
    impact: string;
  }>;
}

const StartupPredictor = () => {
  // Form state - UPDATED TO MATCH API
  const [formData, setFormData] = useState({
    country_code: 'USA',
    region: '',
    city: '',
    category_list: '',
    founded_year: new Date().getFullYear()
  });

  // Category management state
  const [availableCategories, setAvailableCategories] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [categoriesLoading, setCategoriesLoading] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  // UI state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplanationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // API base URL
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Fetch available categories on component mount
  useEffect(() => {
    const fetchCategories = async () => {
      setCategoriesLoading(true);
      try {
        const response = await fetch(`${API_BASE}/categories`);
        if (response.ok) {
          const data = await response.json();
          setAvailableCategories(data.categories);
        }
      } catch (error) {
        console.error('Failed to fetch categories:', error);
      } finally {
        setCategoriesLoading(false);
      }
    };

    fetchCategories();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('year') 
        ? parseInt(value) || 0 
        : value
    }));
  };

  const handleCategoryToggle = (category: string) => {
    setSelectedCategories(prev => 
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const removeCategoryChip = (category: string) => {
    setSelectedCategories(prev => prev.filter(c => c !== category));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    // Add validation at the start of handleSubmit
    if (selectedCategories.length === 0) {
      setError('Please select at least one category');
      setLoading(false);
      return;
    }

    try {
      // Convert array to space-separated string
      const formDataWithCategories = {
        ...formData,
        category_list: selectedCategories.join(' ')
      };

      const response = await fetch(`${API_BASE}/predict/explain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formDataWithCategories)
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data: ExplanationResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getProbabilityColor = (probability: number) => {
    if (probability >= 0.7) return 'text-green-600';
    if (probability >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
       <div className="container-flexible py-3">
          <div className="flex items-center space-x-3">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">ML Startup Success Predictor</h1>
              <p className="text-gray-600">Machine Learning Powered analysis of startup success potential</p>
            </div>
          </div>
        </div>
      </div>

      <div className="container-flexible py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
              <BarChart3 className="h-6 w-6 text-blue-600 mr-2" />
              Company Information
            </h2>

            <div className="space-y-6">
              {/* Geographic Information */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Country Code
                  </label>
                  <select
                    name="country_code"
                    value={formData.country_code}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                    suppressHydrationWarning
                  >
                    <option value="USA">USA</option>
                    <option value="GBR">United Kingdom</option>
                    <option value="CAN">Canada</option>
                    <option value="DEU">Germany</option>
                    <option value="FRA">France</option>
                    <option value="CHN">China</option>
                    <option value="IND">India</option>
                    <option value="ISR">Israel</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Region
                  </label>
                  <input
                    type="text"
                    name="region"
                    value={formData.region}
                    onChange={handleInputChange}
                    placeholder="e.g., SF Bay Area, London, Berlin"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                    suppressHydrationWarning
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  City
                </label>
                <input
                  type="text"
                  name="city"
                  value={formData.city}
                  onChange={handleInputChange}
                  placeholder="e.g., San Francisco, London, Berlin"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                  suppressHydrationWarning
                />
              </div>

              {/* Industry Information - Multi-Select Categories */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Categories {selectedCategories.length === 0 && <span className="text-red-500">*</span>}
                </label>
                
                {/* Selected Categories Chips */}
                {selectedCategories.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-3 p-2 bg-gray-50 rounded-lg">
                    {selectedCategories.map(category => (
                      <span
                        key={category}
                        className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full"
                      >
                        {category}
                        <button
                          type="button"
                          onClick={() => removeCategoryChip(category)}
                          className="ml-2 text-blue-600 hover:text-blue-800"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                
                {/* Dropdown */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white text-left flex items-center justify-between"
                    suppressHydrationWarning
                  >
                    <span className="text-gray-500">
                      {selectedCategories.length === 0 
                        ? 'Select categories...' 
                        : `${selectedCategories.length} selected`}
                    </span>
                    <span className={`transform transition-transform ${dropdownOpen ? 'rotate-180' : ''}`}>
                      ▼
                    </span>
                  </button>
                  
                  {dropdownOpen && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {categoriesLoading ? (
                        <div className="p-3 text-center text-gray-500">Loading categories...</div>
                      ) : (
                        availableCategories.map(category => (
                          <label
                            key={category}
                            className="flex items-center p-3 hover:bg-gray-50 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={selectedCategories.includes(category)}
                              onChange={() => handleCategoryToggle(category)}
                              className="mr-3 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-900 capitalize">
                              {category.replace(/-/g, ' ')}
                            </span>
                          </label>
                        ))
                      )}
                    </div>
                  )}
                </div>
                
                <p className="text-sm text-gray-500 mt-1">
                  Select one or more categories that describe your startup
                  {selectedCategories.length === 0 && (
                    <span className="text-red-500 ml-1">(At least 1 required)</span>
                  )}
                </p>
              </div>

              {/* Temporal Information */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Founded Year
                </label>
                <input
                  type="number"
                  name="founded_year"
                  value={formData.founded_year}
                  onChange={handleInputChange}
                  min="1995"
                  max="2015"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                  suppressHydrationWarning
                />
                <p className="text-sm text-gray-500 mt-1">Model trained on companies founded 1995-2015</p>
              </div>

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                suppressHydrationWarning
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    Analyzing...
                  </span>
                ) : (
                  'Predict Success'
                )}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
              <Lightbulb className="h-6 w-6 text-purple-600 mr-2" />
              Prediction Results
            </h2>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            )}

            {result ? (
              <div className="space-y-6">
                {/* Main Prediction */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
                  <div className="text-center">
                    <div className={`text-4xl font-bold mb-2 ${getProbabilityColor(result.prediction.success_probability)}`}>
                      {(result.prediction.success_probability * 100).toFixed(1)}%
                    </div>
                    <p className="text-gray-600 mb-3">Success Probability</p>
                    <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(result.prediction.confidence)}`}>
                      {result.prediction.confidence.charAt(0).toUpperCase() + result.prediction.confidence.slice(1)} Confidence
                    </div>
                  </div>
                </div>

                {/* Model Info */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">
                    <strong>Model Used:</strong> {result.prediction.model_used.charAt(0).toUpperCase() + result.prediction.model_used.slice(1)}
                  </p>
                  <p className="text-sm text-gray-600">
                    <strong>Prediction:</strong> {result.prediction.prediction === 1 ? 'Likely Success' : 'Likely Failure'}
                  </p>
                </div>

                {/* Top Factors */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Factors</h3>
                  <div className="space-y-2"> {/* Reduced from space-y-3 */}
                    {result.top_factors.map((factor, index) => (
                      <div key={index} className="flex items-center justify-between bg-gray-50 rounded-lg p-2">                        <div className="flex-1">
                          <p className="font-medium text-gray-900">
                            {factor.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            factor.impact === 'positive' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {factor.impact === 'positive' ? '+' : '-'}
                          </span>
                          <span className="text-sm text-gray-600">
                            {Math.abs(factor.importance).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Interpretation */}
                <div className="bg-blue-50 rounded-lg p-3"> {/* Reduced from p-4 */}
                  <h4 className="font-semibold text-blue-900 mb-1 text-sm"> {/* Added text-sm */}
                    Interpretation
                  </h4>
                  <p className="text-blue-800 text-xs"> {/* Reduced from text-sm */}
                    {result.prediction.success_probability > 0.7                       ? "This startup shows strong indicators for success based on historical patterns. Consider factors like market timing and execution quality."
                      : result.prediction.success_probability > 0.4
                      ? "This startup shows mixed signals. Success will likely depend heavily on execution, market conditions, and strategic decisions."
                      : "This startup faces significant challenges based on historical patterns. Consider pivoting or addressing key risk factors."
                    }
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Enter company information and click "Predict Success" to see machine learning model analysis</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 mt-6 text-center text-gray-500 text-sm"> {/* Reduced from mt-12 */}
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

export default StartupPredictor;