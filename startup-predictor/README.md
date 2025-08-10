# *Frontend*

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo GIFs](#demo-gifs)
- [Frontend Technology Stack](#frontend-technology-stack)
- [Frontend Project Structure](#frontend-project-structure)
- [Installation](#installation)
- [Development](#development)
- [Components](#components)
- [API Integration](#api-integration)
- [Styling System](#styling-system)
- [User Experience](#user-experience)
- [Performance](#performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments & References](#acknowledgments--references)

## Overview

The frontend provides an intuitive interface for predicting startup success using machine learning. Users can input company information (location, industry, founding year) and receive instant predictions with detailed explanations powered by SHAP (SHapley Additive exPlanations). A modern, responsive React/Next.js frontend for the Machine Learning Startup Success Predictor. Built with TypeScript, Tailwind CSS, and optimized for user experience with real time predictions, interactive explanations, and an about page with machine learning information, sources used, performance data, etc. 

### Key Results Display
- **Success Probability**: Visual percentage with color coded confidence
- **Model Explanations**: SHAP based feature importance analysis
- **Interactive UI**: Real time form validation and searchable dropdowns
- **Responsive Design**: Mobile first approach with desktop enhancements

## Features

### Core Functionality
- **Real time Predictions** with confidence intervals
- **SHAP Explanations** showing key success factors
- **Interactive Forms** with validation and user feedback
- **Multi category Selection** for industry classification
- **Searchable Dropdowns** for more than 750 regions and cities
- **Responsive Layout** optimized for all devices

### User Interface
- **Modern Design** with gradient backgrounds and glass morphism
- **Smooth Animations** and micro interactions
- **Accessibility** with keyboard navigation and focus states
- **Loading States** and error handling
- **Progressive Enhancement** for better performance

### Data Visualization
- **Probability Meters** with color coded confidence
- **Factor Analysis** showing positive/negative impacts
- **Model Transparency** with algorithm information
- **Historical Context** and interpretation guidance

## Demo GIFs

![demogif1](https://github.com/user-attachments/assets/ec3285e2-7e33-4931-be42-e274dff10c33)

![demogif2](https://github.com/user-attachments/assets/28684aa2-07b7-4d8b-acb0-a51fc770a4e0)

![demogif3](https://github.com/user-attachments/assets/9df4598e-6022-4083-83d1-ebf4b259f331)

![demogifabout](https://github.com/user-attachments/assets/eb975ed2-2f46-49af-ba1c-468d7d6bf746)

## Frontend Technology Stack

### Core Framework
- **React 18** - Component based UI library
- **Next.js 14** - Full stack React framework with App Router
- **TypeScript** - Type safe JavaScript development
- **Tailwind CSS** - Utility first CSS framework

### UI Components & Icons
- **Lucide React** - Beautiful SVG icons
- **Custom Components** - Searchable dropdowns, form controls
- **CSS Grid & Flexbox** - Modern layout systems
- **CSS Custom Properties** - Consistent design tokens

### Development Tools
- **ESLint** - Code linting and formatting
- **TypeScript Config** - Strict type checking
- **Next.js Config** - Build optimization
- **PostCSS** - CSS processing and optimization

## Frontend Project Structure

```
startup-predictor/
├── .next/                      # Next.js build output (auto-generated)
├── app/                        # Next.js App Router
│   ├── about/
│   │   └── page.tsx            # About page with project details
│   ├── globals.css             # Global styles and CSS reset
│   ├── layout.tsx              # Root layout component
│   ├── page.tsx                # Main prediction interface
│   └── favicon.ico             # Application favicon
├── node_modules/               # Package dependencies (auto-generated)
├── public/                     # Static assets
│   ├── file.svg                # File icon
│   ├── globe.svg               # Globe icon  
│   ├── next.svg                # Next.js logo
│   ├── vercel.svg              # Vercel logo
│   └── window.svg              # Window icon
├── styles/
│   └── app.css                 # Custom styles and design system
├── .gitignore                  # Git ignore rules
├── eslint.config.mjs           # ESLint configuration
├── next-env.d.ts               # Next.js TypeScript declarations
├── next.config.ts              # Next.js configuration (TypeScript)
├── package-lock.json           # NPM lock file
├── package.json                 # Dependencies and scripts  
├── postcss.config.mjs           # PostCSS configuration
├── README.md                    # Frontend documentation
└── tsconfig.json                # TypeScript configuration
```

## Installation

### Prerequisites
- **Node.js 16+**
- **npm** 
- **Backend API** running on `http://localhost:8000` (see [Backend README](../src/README.md))

### Setup Steps

1. **Navigate to Frontend Directory**
   ```bash
   cd startup-predictor
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Environment Configuration**
   ```bash
   # Create environment file
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   ```

4. **Start Development Server**
   ```bash
   npm run dev
   ```

5. **Access Application**
   Open [http://localhost:3000](http://localhost:3000) in your browser

## Development

### Available Scripts

```bash
# Development server with hot reload
npm run dev

# Production build
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint
```

### Development Workflow

1. **Code Structure**: Follow Next.js App Router conventions
2. **TypeScript**: Use strict typing for all components and interfaces
3. **Components**: Create reusable, accessible components
4. **Styling**: Use Tailwind classes with custom CSS for complex designs
5. **Testing**: Manual testing across devices and browsers

### Environment Variables

```bash
# .env.local (optional)
NEXT_PUBLIC_API_URL=http://localhost:8000  # Backend API URL
```

## Components

### Main Components

#### `StartupPredictor` (page.tsx)
The core application component containing:

**State Management**
```typescript
// Form data matching API requirements
const [formData, setFormData] = useState({
  country_code: 'USA',
  region: '',
  city: '',
  category_list: '',
  founded_year: new Date().getFullYear()
});

// Dropdown data and loading states
const [regions, setRegions] = useState<string[]>([]);
const [cities, setCities] = useState<string[]>([]);
const [regionsLoading, setRegionsLoading] = useState(false);

// Results and UI state
const [result, setResult] = useState<ExplanationResponse | null>(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
```

**Key Features**
- Form validation with real time feedback
- API integration with error handling
- Results visualization with SHAP explanations
- Responsive layout with mobile optimization

#### `SearchableDropdown`
Advanced dropdown component with:

**Features**
- **Search Functionality**: Filter options with case-insensitive search
- **Keyboard Navigation**: Arrow keys, Enter, Escape support
- **Accessibility**: ARIA labels and focus management
- **Loading States**: Visual indicators during data fetch
- **Clear Functionality**: Easy option clearing

**Usage Example**
```typescript
<SearchableDropdown
  options={regions}
  value={formData.region}
  onChange={(value) => handleDropdownChange('region', value)}
  placeholder="Search regions..."
  loading={regionsLoading}
  required={true}
  name="region"
/>
```

#### `AboutPage` (about/page.tsx)
Comprehensive project documentation including:
- Academic foundation and methodology
- Technology stack with visual indicators
- Performance metrics and model details
- Links to notebooks and GitHub repository
- Visual examples and feature highlights

### Reusable UI Patterns

#### Form Controls
- **Input Fields**: Enhanced with focus states and validation
- **Select Dropdowns**: Styled with consistent theming
- **Multi-select**: Category chips with remove functionality
- **Labels**: Clear hierarchy with required field indicators

#### Visual Feedback
- **Loading States**: Spinners and skeleton screens
- **Error Messages**: Clear, actionable error display
- **Success Indicators**: Color coded probability displays
- **Confidence Badges**: High/Medium/Low confidence visualization

## API Integration

### Endpoint Integration

#### Prediction with Explanations
```typescript
const response = await fetch(`${API_BASE}/predict/explain`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    country_code: 'USA',
    region: 'SF Bay Area',
    city: 'San Francisco',
    category_list: 'software mobile',
    founded_year: 2010
  })
});
```

#### Response Handling
```typescript
interface ExplanationResponse {
  prediction: {
    success_probability: number;
    prediction: number;
    model_used: string;
    confidence: string;
  };
  feature_importance: Record<string, number>;
  top_factors: Array<{
    feature: string;
    importance: number;
    impact: string;
  }>;
}
```

### Data Fetching

#### Dropdown Options
- **Regions**: 750+ global regions loaded on mount
- **Cities**: Comprehensive city database
- **Categories**: 15 industry categories with multi select

#### Error Handling
- Network errors with retry suggestions
- Validation errors with field-specific feedback
- API errors with user friendly messages
- Loading timeouts with fallback options

## Styling System

### Design Tokens (CSS Custom Properties)

```css
:root {
  /* Primary Colors */
  --primary-blue: #2563eb;
  --primary-blue-dark: #1d4ed8;
  --secondary-purple: #7c3aed;
  
  /* Semantic Colors */
  --accent-green: #059669;
  --accent-red: #dc2626;
  
  /* Gray Scale */
  --gray-50: #f8fafc;
  --gray-900: #0f172a;
  
  /* Effects */
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  --border-radius: 0.75rem;
  --transition: 0.2s ease;
}
```

### Visual Design Features

#### Glass Morphism Effects
- **Backdrop Blur**: Subtle blur effects on cards and modals
- **Transparency**: Semi-transparent backgrounds with depth
- **Border Highlights**: Gradient borders for visual hierarchy

#### Micro-interactions
- **Hover States**: Smooth transitions and elevation changes
- **Focus Indicators**: Clear focus rings for accessibility
- **Loading Animations**: Smooth spinners and progress indicators
- **Card Animations**: Subtle lift effects on interaction

#### Responsive Breakpoints
```css
/* Mobile First Approach */
@media (min-width: 768px)  { /* Tablet */ }
@media (min-width: 1024px) { /* Desktop */ }
@media (min-width: 1400px) { /* Large Desktop */ }
```

### Component Styling

#### Cards
- **Background**: Semi transparent with blur
- **Borders**: Subtle with gradient accents
- **Shadows**: Multi layer depth effects
- **Hover Effects**: Smooth elevation changes

#### Forms
- **Input Focus**: Highlighted borders with glow
- **Validation**: Color coded feedback states
- **Placeholders**: Helpful hint text
- **Disabled States**: Clear visual indicators

## User Experience

### Interaction Flow

1. **Landing**: User sees clean, professional interface
2. **Form Filling**: Guided input with validation and hints
3. **Submission**: Loading states with progress indication
4. **Results**: Immediate feedback with visual explanations
5. **Exploration**: Interactive elements for deeper insights

### Accessibility Features

#### Keyboard Navigation
- **Tab Order**: Logical flow through form elements
- **Arrow Keys**: Dropdown navigation support
- **Enter/Space**: Action triggers and selections
- **Escape**: Modal/dropdown dismissal

#### Screen Reader Support
- **ARIA Labels**: Descriptive labels for all inputs
- **Semantic HTML**: Proper heading hierarchy
- **Focus Management**: Logical focus flow
- **Status Updates**: Announcements for dynamic content

#### Visual Accessibility
- **Color Contrast**: WCAG AA compliant ratios
- **Focus Indicators**: Clear visual focus states
- **Font Sizes**: Readable text at all zoom levels
- **Reduced Motion**: Respects user preferences

### Error Handling

#### Form Validation
```typescript
// Real-time validation with helpful messages
if (selectedCategories.length === 0) {
  setError('Please select at least one category');
  return;
}

if (!formData.region.trim()) {
  setError('Please select a region');
  return;
}
```

#### API Error Management
- **Network Errors**: Clear retry instructions
- **Validation Errors**: Field specific guidance
- **Server Errors**: Friendly error messages
- **Timeout Handling**: Graceful degradation

## Performance

### Optimization Strategies

#### Bundle Optimization
- **Code Splitting**: Automatic route based splitting
- **Tree Shaking**: Unused code elimination
- **Image Optimization**: Next.js automatic optimization
- **Font Loading**: Optimized Google Fonts integration

#### Runtime Performance
- **React Optimization**: Proper key usage and memo
- **State Management**: Efficient state updates
- **API Caching**: Smart data fetching strategies
- **Lazy Loading**: Component and route lazy loading

### UX Performance
- **Loading States**: Immediate visual feedback
- **Progressive Enhancement**: Core functionality first
- **Skeleton Screens**: Content placeholders
- **Error Boundaries**: Graceful error handling

## Deployment

### Build Process

```bash
# Create optimized production build
npm run build

# Verify build locally
npm start
```

### Deployment Options

#### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel --prod
```

#### Static Export
```bash
# For static hosting
npm run build
npm run export
```

#### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Configuration

#### Production Variables
```bash
# .env.production
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_ANALYTICS_ID=your-analytics-id
```

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
