import pandas as pd
import os

def extract_dropdown_options(raw_data_path='data/raw/startups_data.csv', 
                           output_dir='data/processed/'):
    """
    Extracts unique regions and cities for UI dropdowns from raw startup data
        raw_data_path: Path to the raw startups data CSV
        output_dir: Directory to save the output CSV files
        Returns tuple: (unique_regions_list, unique_cities_list)
    """
    
    # Loads just the geographic columns (only 3 columns exist)
    df_geo = pd.read_csv(raw_data_path, 
                         encoding='latin-1', 
                         usecols=['region', 'city', 'founded_year'])
    
    # Applies the same temporal filter used in preprocessing (1995-2015)
    df_geo = df_geo[(df_geo['founded_year'] >= 1995) & (df_geo['founded_year'] <= 2015)]
    
    # Gets all unique regions and cities (excludes NaN values)
    unique_regions = df_geo['region'].dropna().unique()
    unique_cities = df_geo['city'].dropna().unique()
    
    # Creatse output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames and saves as CSV
    regions_df = pd.DataFrame({'region': sorted(unique_regions)})
    cities_df = pd.DataFrame({'city': sorted(unique_cities)})
    
    regions_df.to_csv(os.path.join(output_dir, 'unique_regions.csv'), index=False)
    cities_df.to_csv(os.path.join(output_dir, 'unique_cities.csv'), index=False)
    
    print(f"Saved {len(unique_regions)} unique regions to {output_dir}/unique_regions.csv")
    print(f"Saved {len(unique_cities)} unique cities to {output_dir}/unique_cities.csv")
    
    return sorted(unique_regions.tolist()), sorted(unique_cities.tolist())

if __name__ == "__main__":
    regions, cities = extract_dropdown_options()
    print(f"\nFirst 10 regions: {regions[:10]}")
    print(f"First 10 cities: {cities[:10]}")