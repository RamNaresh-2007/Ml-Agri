import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from features import get_preprocessor

def generate_plots():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'processed_crop_yield.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'model.joblib')
    preprocessor_path = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')
    results_dir = os.path.join(BASE_DIR, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    print("Generating Yield Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Yield'], bins=50, kde=True, color='green')
    plt.title('Distribution of Crop Yield')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(results_dir, 'yield_distribution.png'))
    plt.close()
    
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print("Generating Actual vs Predicted Plot...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    X = df.drop(columns=['Yield'])
    y = df['Yield']
    
    # We use a test split to evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Crop Yield')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
    plt.close()

    print("Generating Feature Importance Plot...")
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        
        # Sort and take top 20 for readability
        indices = importances.argsort()[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_importances, y=top_features, palette='viridis')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'feature_importances.png'))
        plt.close()
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    print("All plots generated successfully in the 'results' directory.")

if __name__ == "__main__":
    generate_plots()
