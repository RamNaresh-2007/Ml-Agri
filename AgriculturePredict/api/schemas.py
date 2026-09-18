from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    Crop: str = Field(..., example="Rice")
    Crop_Year: int = Field(..., example=2020)
    Season: str = Field(..., example="Kharif")
    State: str = Field(..., example="Assam")
    Area: float = Field(..., example=1000.0)
    Annual_Rainfall: float = Field(..., example=2000.0)
    Fertilizer: float = Field(..., example=50000.0)
    Pesticide: float = Field(..., example=1500.0)

class PredictBatchRequest(BaseModel):
    data: List[PredictRequest]

class PredictResponse(BaseModel):
    predicted_yield: float

class PredictBatchResponse(BaseModel):
    predictions: List[float]
