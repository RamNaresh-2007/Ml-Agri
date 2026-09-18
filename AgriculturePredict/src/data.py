import pandas as pd
import os

def load_data(filepath):
    print(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    print("Cleaning data...")
    # Drop 'Production' to prevent data leakage (since Yield = Production/Area)
    if 'Production' in df.columns:
        df = df.drop(columns=['Production'])
    
    # Handle missing values (simple drop for this example)
    initial_shape = df.shape
    df = df.dropna()
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values.")
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    return df

def save_data(df, filepath):
    print(f"Saving processed data to {filepath}")
    df.to_csv(filepath, index=False)
    print("Done.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(BASE_DIR, 'data', 'crop_yield.csv')
    processed_data_path = os.path.join(BASE_DIR, 'data', 'processed_crop_yield.csv')
    
    df = load_data(raw_data_path)
    df_clean = clean_data(df)
    save_data(df_clean, processed_data_path)
