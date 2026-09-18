import requests
import json

url_single = "http://127.0.0.1:8000/predict"
payload_single = {
    "Crop": "Rice",
    "Crop_Year": 2020,
    "Season": "Kharif",
    "State": "Assam",
    "Area": 1000.0,
    "Annual_Rainfall": 2000.0,
    "Fertilizer": 50000.0,
    "Pesticide": 1500.0
}

print("Testing Single Prediction:")
response = requests.post(url_single, json=payload_single)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

url_batch = "http://127.0.0.1:8000/predict_batch"
payload_batch = {
    "data": [
        payload_single,
        {
            "Crop": "Wheat",
            "Crop_Year": 2021,
            "Season": "Rabi",
            "State": "Punjab",
            "Area": 2000.0,
            "Annual_Rainfall": 500.0,
            "Fertilizer": 100000.0,
            "Pesticide": 3000.0
        }
    ]
}

print("Testing Batch Prediction:")
response_batch = requests.post(url_batch, json=payload_batch)
print(f"Status: {response_batch.status_code}")
print(f"Response: {response_batch.json()}")
