import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any

# Sets up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StartupDataProcessor:
    """
    Preprocessing pipeline that replicates methodology
    from 02_data_preprocessing_feature_engineering notebook
    """
    
    def __init__(self):
        self.region_density_mapping = {}
        self.city_density_mapping = {}
        self.top_categories = [
            'software', 'mobile', 'social', 'media', 'web', 'e-commerce', 
            'biotechnology', 'curated', 'health', 'advertising', 'games', 
            'enterprise', 'technology', 'marketing', 'analytics'
        ]
        self.scaler = StandardScaler()
        self.founding_year_mean = None
        self.founding_year_std = None
        self.feature_columns = []
        
    def clean_and_extract_categories(self, category_string: str) -> List[str]:
        """
        Cleans categories and handle parsing artifacts
        """
        if pd.isna(category_string) or category_string == "":
            return []
        
        # Remove pipes, split, and clean
        categories = category_string.replace('|', ' ').split()
        cleaned_categories = []
        
        for cat in categories:
            cat = cat.strip()
            # Removes parsing artifacts found in EDA
            if cat and cat not in ['and', '&', '', ' ']:
                cleaned_categories.append(cat.lower())  # Standardize case
                
        return cleaned_categories
    
    def create_density_tiers(self, counts_series: pd.Series, n_tiers: int = 5) -> pd.Series:
        """
        Creates density tiers based on startup counts
        """
        if len(counts_series) == 0:
            return pd.Series(dtype='int64')
        
        # Uses quantile based binning for more balanced tiers
        tiers = pd.qcut(counts_series.rank(method='first'),
                       q=n_tiers, labels=range(1, n_tiers+1), duplicates='drop')
        return tiers
    
    def assign_economic_era(self, founded_year: float) -> str:
        """
        Assigns economic era based on founding year
        """
        if pd.isna(founded_year):
            return 'recovery'  # Default to most common era
        elif 1995 <= founded_year <= 2000:
            return 'dotcom_era'
        elif 2001 <= founded_year <= 2008:
            return 'post_crash'
        elif 2009 <= founded_year <= 2015:
            return 'recovery'
        else:
            return 'recovery'  # Default for out-of-range years
    
    def fit(self, df: pd.DataFrame) -> 'StartupDataProcessor':
        """
        Fits the preprocessor on training data
        Learns all mappings and transformations
        """
        logger.info("Fitting StartupDataProcessor on training data...")
        
        # Creates geographic density mappings
        region_counts = df['region'].value_counts()
        self.region_density_mapping = self.create_density_tiers(region_counts, n_tiers=5)
        
        city_counts = df['city'].value_counts()
        self.city_density_mapping = self.create_density_tiers(city_counts, n_tiers=5)
        
        logger.info(f"Created region density mapping for {len(self.region_density_mapping)} regions")
        logger.info(f"Created city density mapping for {len(self.city_density_mapping)} cities")
        
        # Calculates founding year statistics for standardization
        self.founding_year_mean = df['founded_year'].mean()
        self.founding_year_std = df['founded_year'].std()
        
        logger.info(f"Founding year statistics - Mean: {self.founding_year_mean:.1f}, Std: {self.founding_year_std:.1f}")
        
        # Defines expected feature columns after transformation (MATCHES MODEL EXACTLY)
        self.feature_columns = [
            # Geographic features (3)
            'region_startup_density',
            'city_startup_density', 
            'is_usa',
            # Industry features (15) - MUST MATCH YOUR MODEL'S ORDER
            'category_software', 'category_mobile', 'category_social', 'category_media',
            'category_web', 'category_e-commerce', 'category_biotechnology', 'category_curated',
            'category_health', 'category_advertising', 'category_games', 'category_enterprise',
            'category_technology', 'category_marketing', 'category_analytics',
            # Temporal features (4)
            'founded_year_std',
            'era_dotcom_era',
            'era_post_crash', 
            'era_recovery'
        ]
        
        logger.info(f"Expected {len(self.feature_columns)} features after preprocessing")
        return self
    
    def transform_single(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single startup record for API predictions
        """
        # Converts to DataFrame for consistent processing
        df = pd.DataFrame([data])
        return self.transform(df)[0]
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms data using fitted mappings
        """
        logger.info(f"Transforming {len(df)} startup records...")
        
        # Creates a copy to AVOID modifying original *IMPORTANT*
        df_processed = df.copy()
        
        # 1. GEOGRAPHIC FEATURE ENGINEERING
        # Region startup density (5 tier ranking system)
        df_processed['region_startup_density'] = df_processed['region'].map(self.region_density_mapping)
    
        # Fill unknown regions with tier 5 (lowest density)
        df_processed['region_startup_density'].fillna(5, inplace=True)
        
        # City startup density (5 tier ranking system) 
        df_processed['city_startup_density'] = df_processed['city'].map(self.city_density_mapping)
        
        # Fill unknown cities with tier 5 (lowest density)
        df_processed['city_startup_density'].fillna(5, inplace=True)
        
        # USA binary flag
        df_processed['is_usa'] = (df_processed['country_code'] == 'USA').astype(int)
        
        # 2. INDUSTRY FEATURE ENGINEERING
        # Cleans and extract categories
        df_processed['categories_clean'] = df_processed['category_list'].apply(self.clean_and_extract_categories)
        
        # Creates binary features for top categories
        for category in self.top_categories:
            df_processed[f'category_{category}'] = df_processed['categories_clean'].apply(
                lambda x: 1 if category in x else 0
            )
        
        # 3. TEMPORAL FEATURE ENGINEERING
        # Standardized founding year
        df_processed['founded_year_std'] = (
            df_processed['founded_year'] - self.founding_year_mean
        ) / self.founding_year_std
        
        # Economic era classification
        df_processed['economic_era'] = df_processed['founded_year'].apply(self.assign_economic_era)
        
        # Creates era dummy variables
        era_dummies = pd.get_dummies(df_processed['economic_era'], prefix='era')
        df_processed = pd.concat([df_processed, era_dummies], axis=1)
        
        # Ensures all expected era columns exist
        for era in ['era_dotcom_era', 'era_post_crash', 'era_recovery']:
            if era not in df_processed.columns:
                df_processed[era] = 0
        
        # Handles unknown economic era
        if 'era_unknown' in df_processed.columns:
            # Distributes unknown cases to recovery era (most common)
            unknown_mask = df_processed['era_unknown'] == 1
            df_processed.loc[unknown_mask, 'era_recovery'] = 1
            df_processed.loc[unknown_mask, 'era_unknown'] = 0
        
        # 5. FEATURE SELECTION - EXTRACT FEATURES MODEL EXPECTS
        try:
            feature_matrix = df_processed[self.feature_columns].values
        except KeyError as e:
            missing_cols = [col for col in self.feature_columns if col not in df_processed.columns]
            available_cols = list(df_processed.columns)
            logger.error(f"Missing columns after preprocessing: {missing_cols}")
            logger.error(f"Available columns: {available_cols}")
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # Handles any remaining NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        logger.info(f"Transformation complete. Output shape: {feature_matrix.shape}")
        logger.info(f"Feature order: {self.feature_columns}")
        return feature_matrix
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit on data and transform it
        """
        return self.fit(df).transform(df)
    
    def save(self, filepath: str) -> None:
        """
        Saves the fitted preprocessor to disk
        """
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StartupDataProcessor':
        """
        Loads a fitted preprocessor from disk
        """
        processor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return processor

def create_and_fit_preprocessor(training_data_path: str, encoding: str = 'utf-8') -> StartupDataProcessor:
    """
    Creates and fit preprocessor on training data
    """
    logger.info(f"Loading training data from {training_data_path}")
    
    # Loads the training data with specified encoding
    df = pd.read_csv(training_data_path, encoding=encoding)
    
    # Creates and fit preprocessor
    processor = StartupDataProcessor()
    processor.fit(df)
    
    return processor

def preprocess_single_input(data: Dict[str, Any], processor: StartupDataProcessor) -> np.ndarray:
    """
    Preprocess a single startup input for API predictions
        data: Dictionary with startup features
        processor: Fitted StartupDataProcessor
        Returns preprocessed feature array ready for model prediction
    """
    return processor.transform_single(data)

# Example usage for API integration
def load_production_preprocessor(model_dir: str = "models") -> StartupDataProcessor:
    """
    Loads the production preprocessor for API use
    """
    preprocessor_path = Path(model_dir) / "preprocessor.pkl"
    
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    
    return StartupDataProcessor.load(str(preprocessor_path))

if __name__ == "__main__":
    # Example of how to create and save the preprocessor
    # This would be run once during model training
    
    # Creates preprocessor from training data
    processor = create_and_fit_preprocessor("data/processed/startup_data_processed.csv")
    
    # Saves for production use
    processor.save("models/preprocessor.pkl")
    
    # Tests with sample data - UPDATED TO MATCH API INPUTS
    sample_data = {
        'country_code': 'USA',
        'region': 'SF Bay Area', 
        'city': 'San Francisco',
        'category_list': 'software enterprise',  # Space-separated instead of pipe-separated
        'founded_year': 2010
    }
    
    # Transforms sample
    features = processor.transform_single(sample_data)
    print(f"Preprocessed features shape: {features.shape}")
    print(f"Sample features: {features[:5]}...")  # First 5 features