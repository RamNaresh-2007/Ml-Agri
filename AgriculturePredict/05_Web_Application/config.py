import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Dataset paths
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield.csv")
SAMPLE_DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "crop_yield_sample.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_DIR, "01_Datasets", "processed_crop_yield.csv")

# Model & Outputs paths
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "06_Outputs_and_Utils")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "Model_Outputs", "Regression")
CLUSTERING_DIR = os.path.join(OUTPUTS_DIR, "Model_Outputs", "Clustering")
PROCESSED_DATA_DIR = os.path.join(OUTPUTS_DIR, "processed_data")
EDA_PLOTS_DIR = os.path.join(PROJECT_DIR, "03_Exploratory_Data_Analysis", "plots")
EDA_OUTPUTS_DIR = os.path.join(PROJECT_DIR, "03_Exploratory_Data_Analysis", "EDA_Outputs")

# Model files
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessor.joblib")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "outputs", "metrics.json")
CLUSTERING_SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "outputs", "co4_clustering_summary.json")

# Ensure fallback within 06_Outputs_and_Utils if random_forest.joblib is named differently
if not os.path.exists(MODEL_PATH):
    alt_model = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(alt_model):
        MODEL_PATH = alt_model

# Column Definitions
CATEGORICAL_COLS = ["Crop", "Season", "State"]
NUMERIC_COLS = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
TARGET_COL = "Yield"
LEAKAGE_COL = "Production"

# Additional directories & aliases matching reference ML architecture
LOGISTIC_DIR = os.path.join(OUTPUTS_DIR, "outputs", "logistic")
HIERARCHICAL_DIR = os.path.join(OUTPUTS_DIR, "outputs", "hierarchical")
DBSCAN_DIR = os.path.join(OUTPUTS_DIR, "outputs", "dbscan")
SPLITS_DIR = os.path.join(OUTPUTS_DIR, "splits")
PROCESSED_DIR = os.path.join(OUTPUTS_DIR, "processed")
SCALING_RESULTS_DIR = os.path.join(OUTPUTS_DIR, "Outputs")

RANDOM_STATE = 42
NUMERIC_COLUMNS = NUMERIC_COLS
TARGET_COLUMN = TARGET_COL

# Web App Configuration
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
SECRET_KEY = "agri-yield-secret-key-2026"

