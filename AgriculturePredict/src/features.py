import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

def get_preprocessor():
    # Define categorical and numerical columns based on crop_yield dataset
    categorical_cols = ['Crop', 'Season', 'State']
    # Removing 'Crop_Year' from scaling as it can be treated as numeric or categorical. Let's treat it as numeric.
    numerical_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine transformers in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, numerical_cols, categorical_cols

if __name__ == "__main__":
    # Test preprocessor on data
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(BASE_DIR, 'data', 'processed_crop_yield.csv')
    
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=['Yield'])
    y = df['Yield']
    
    preprocessor, _, _ = get_preprocessor()
    print("Fitting preprocessor...")
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed data shape: {X_transformed.shape}")
    
    # We will save the fitted preprocessor in train.py after splitting the data 
    # to avoid data leakage (fitting on test data).
    print("Feature engineering module ready.")
