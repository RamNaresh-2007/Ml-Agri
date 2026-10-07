"""
Shared Utilities & Data Access Helpers for AgriculturePredict.
Provides unified data loading, cleaning, metrics calculation, and persona classification.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "05_Web_Application"))
try:
    import config
except ImportError:
    import importlib.util
    config_path = os.path.join(PROJECT_DIR, "05_Web_Application", "config.py")
    if os.path.exists(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        config = None


def load_raw():
    """Return untouched raw crop yield dataset."""
    return pd.read_csv(config.RAW_DATA_PATH)


def clean_data(df=None, save=True):
    """
    Cleans dataset:
      - Drops 'Production' to prevent target leakage (Yield = Production / Area).
      - Drops rows with null values.
      - Strips whitespace from categorical strings.
      - Ensures numeric types.
    """
    if df is None:
        df = load_raw()
    df = df.copy()

    if config.LEAKAGE_COL in df.columns:
        df = df.drop(columns=[config.LEAKAGE_COL])

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()

    for col in config.NUMERIC_COLS + [config.TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)

    if save:
        os.makedirs(os.path.dirname(config.CLEANED_DATA_PATH), exist_ok=True)
        df.to_csv(config.CLEANED_DATA_PATH, index=False)

    return df, dropped


_CLEANED_CACHE = None

def load_cleaned(force_reload=False):
    """Return cleaned dataset, using in-memory cache and generating on first call if missing."""
    global _CLEANED_CACHE
    if _CLEANED_CACHE is not None and not force_reload:
        return _CLEANED_CACHE
    if not os.path.exists(config.CLEANED_DATA_PATH):
        df, _ = clean_data()
    else:
        df = pd.read_csv(config.CLEANED_DATA_PATH)
    _CLEANED_CACHE = df
    return _CLEANED_CACHE


def load_model_and_preprocessor():
    """Load the trained machine learning pipeline / model and preprocessor."""
    model = None
    preprocessor = None
    if os.path.exists(config.MODEL_PATH):
        model = joblib.load(config.MODEL_PATH)
    if os.path.exists(config.PREPROCESSOR_PATH):
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    return model, preprocessor


def get_metadata(df=None):
    """Extract metadata, available categories, and statistics."""
    if df is None:
        df = load_cleaned()
        
    crops = sorted(df['Crop'].dropna().unique().tolist())
    states = sorted(df['State'].dropna().unique().tolist())
    seasons = sorted(df['Season'].dropna().unique().tolist())

    return {
        "total_records": len(df),
        "crops": crops,
        "crops_count": len(crops),
        "states": states,
        "states_count": len(states),
        "seasons": seasons,
        "seasons_count": len(seasons),
        "avg_yield": round(float(df["Yield"].mean()), 2) if "Yield" in df.columns else 0,
        "avg_rainfall": round(float(df["Annual_Rainfall"].mean()), 1) if "Annual_Rainfall" in df.columns else 0,
        "total_area": round(float(df["Area"].sum()), 0) if "Area" in df.columns else 0
    }


def compute_regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }


def get_agro_persona(rainfall, fertilizer, area):
    """
    Classify farming scenario into Agro-Ecological Archetypes / Personas
    derived from K-Means and Hierarchical Clustering:
      - Cluster 1: High-Yield Intensive Agro-Zone (High inputs, irrigated/fertile)
      - Cluster 2: Rainfed Traditional Belt (Moderate rainfall & inputs)
      - Cluster 3: Arid / Low-Intensity Subsistence (Low moisture, smallholder)
    """
    if rainfall >= 1800 and fertilizer >= 500000:
        return {
            "name": "High-Yield Intensive Agro-Zone",
            "cluster_id": 1,
            "badge": "Intensive Fertile Zone",
            "color": "#10b981",
            "desc": "High input efficiency, abundant water availability, and optimal soil nutrient replenishment."
        }
    elif rainfall < 800 or fertilizer < 50000:
        return {
            "name": "Arid / Rain-Deficit Agro-Zone",
            "cluster_id": 3,
            "badge": "Moisture-Stressed Zone",
            "color": "#ef4444",
            "desc": "Sub-optimal precipitation and scarce fertilization; vulnerable to droughts."
        }
    else:
        return {
            "name": "Rainfed Balanced Agro-Zone",
            "cluster_id": 2,
            "badge": "Standard Rainfed Zone",
            "color": "#38bdf8",
            "desc": "Representative moderate-intensity cultivation zone with balanced regional rainfall."
        }


def generate_agronomic_recommendations(crop, rainfall, fertilizer, area, pred_yield):
    """Generate dynamic, actionable AI agronomic coaching tips."""
    tips = []
    
    # Rainfall feedback
    if rainfall < 700:
        tips.append(f"💧 Moisture Alert: Annual rainfall ({rainfall} mm) is low. Implement drip or micro-irrigation to prevent moisture stress.")
    elif rainfall > 2500:
        tips.append(f"🌊 Drainage Advisory: High rainfall ({rainfall} mm) detected. Ensure raised-bed furrows to mitigate waterlogging and root rot.")
    else:
        tips.append(f"🌧️ Optimal Rainfall: Precipitation ({rainfall} mm) is well aligned with regional agro-climatic demands.")

    # Fertilizer intensity
    if area > 0:
        fert_per_ha = fertilizer / area
        if fert_per_ha > 300:
            tips.append(f"🌱 Fertilizer Intensity: {fert_per_ha:.1f} kg/ha is high. Split application into basal and top-dressing to prevent nutrient leaching.")
        elif fert_per_ha < 40:
            tips.append(f"🧪 Nutrient Deficit: Fertilizer application ({fert_per_ha:.1f} kg/ha) is low. Consider bio-fertilizers or balanced NPK supplementation.")
        else:
            tips.append(f"✅ Balanced Nutrition: Fertilizer dosage ({fert_per_ha:.1f} kg/ha) falls in the sustainable efficiency envelope.")

    # Yield performance
    if pred_yield > 50:
        tips.append(f"🏆 High Yield Outlook: Projected yield ({pred_yield:.2f}) indicates superior productivity potential. Verify storage and market linkages.")
    else:
        tips.append(f"📈 Yield Enhancement: Expected yield is {pred_yield:.2f}. Integrating pest scouting and hybrid seed varieties can unlock +15-20% gain.")

    return tips


if __name__ == "__main__":
    print("=" * 60)
    print("  AGRIYIELD AI UTILITIES CHECK")
    print("=" * 60)
    data = load_cleaned()
    meta = get_metadata(data)
    print(f"Cleaned dataset loaded: {meta['total_records']} rows, {meta['crops_count']} crops, {meta['states_count']} states.")
    model, prep = load_model_and_preprocessor()
    print(f"Model loaded: {'Yes' if model is not None else 'No'} | Preprocessor loaded: {'Yes' if prep is not None else 'No'}")
    print("Utils module verified successfully.")
