from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from json import dumps
from uvicorn import run
import os
import numpy as np

from typing import List
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow import expand_dims
from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array
from pydantic import BaseModel

app = FastAPI()

new_model = tf.keras.models.load_model('c2_N35_cell2.h5', compile=False)
new_model.compile()
new_model.summary()


# Mapping dictionary
class_mapping = {
    0: "Stationary-Or-Eating",
    1: "sleeping",
    2: "walking"
}
class InputData(BaseModel):
    data: list[list[float]]

class_predictions = array([
    'Stationary',
    'Sleeping'
])

@app.get("/")
async def root():
    return {"message":"Version 0.0.1"}

# @app.post("/predict/")
# async def predict(data: InputData):
#     if data == "":
#         return {"message": "No data provided"}
    
#     input_array = np.array(data.data)
#     predictions = new_model.predict(input_array)
#     score = softmax(predictions[0])
    
#     class_prediction = class_predictions[argmax(score)]
#     model_score = round(max(score) * 100, 2)

#     return {
#         "model-prediction": class_prediction,
#         "model-prediction-confidence-score": model_score
#     }
#     # return {"result": "Welcome!!!"}

# @app.post("/predict_behavior")
# def predict_behavior(data: InputData):
#     # Prepare input data for prediction
#     input_array = np.array(data.data)
    
#     # Make prediction
#     predictions = new_model.predict(input_array)

#     # Get predicted class indices
#     prediction_class_indices = np.argmax(predictions, axis=1)

#     # Map class indices to class names
#     prediction_class_names = [class_mapping.get(idx, "Unknown") for idx in prediction_class_indices]

#     return {"predicted_behavior": prediction_class_names}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)