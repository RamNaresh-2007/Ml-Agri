from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd
import os
import time
from .schemas import PredictRequest, PredictBatchRequest, PredictResponse, PredictBatchResponse

app = FastAPI(title="AgriculturePredict API", description="API for predicting crop yield.")

# Load model and preprocessor on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

model = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    print("Loading model and preprocessor...")
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# Monitoring hook: simple middleware to log processing time
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Process Time: {process_time:.4f}s")
    return response

@app.post("/predict", response_model=PredictResponse)
def predict_yield(request: PredictRequest):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")
    
    # Convert request to DataFrame
    df = pd.DataFrame([request.dict()])
    
    # Preprocess
    try:
        X_transformed = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")
    
    # Predict
    try:
        prediction = model.predict(X_transformed)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    print(f"Single Prediction Inputs: {request.dict()} | Output: {prediction}")
    
    return PredictResponse(predicted_yield=prediction)

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_yield_batch(request: PredictBatchRequest):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")
    
    # Convert request to DataFrame
    df = pd.DataFrame([item.dict() for item in request.data])
    
    # Preprocess
    try:
        X_transformed = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")
    
    # Predict
    try:
        predictions = model.predict(X_transformed).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    print(f"Batch Prediction Count: {len(predictions)}")
    
    return PredictBatchResponse(predictions=predictions)
