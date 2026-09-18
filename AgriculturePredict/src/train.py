import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from features import get_preprocessor

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(BASE_DIR, 'data', 'processed_crop_yield.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'model.joblib')
    preprocessor_path = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

    print("Loading processed data...")
    df = pd.read_csv(processed_data_path)
    
    # Drop rows with NaN in target or features if any slipped through
    df = df.dropna()

    X = df.drop(columns=['Yield'])
    y = df['Yield']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Retrieving and fitting preprocessor (Feature Store)...")
    preprocessor, _, _ = get_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("Saving preprocessor...")
    joblib.dump(preprocessor, preprocessor_path)

    print("Training RandomForestRegressor...")
    # Using small n_estimators for speed, can be increased for better performance
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_transformed, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_transformed)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print("Saving trained model...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    print("Training complete.")

if __name__ == "__main__":
    train_model()
