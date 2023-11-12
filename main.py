from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from json import dumps
from uvicorn import run
import os
import numpy as np

app = FastAPI()

new_model = tf.keras.models.load_model('c2_N35_cell2.h5', compile=False)
new_model.compile()
new_model.summary()
dataframes = []

@app.get("/")
async def root():
    return {"message":"Version 0.0.1"}

@app.post("/predict/")
async def predict(data: str=dataframes):
#     # if data == "":
#     #     return {"message": "No data provided"}
    
#     # predictions = new_model.predict(input_array)
#     # score = softmax(predictions[0])
    
#     # class_prediction = class_predictions[argmax(score)]
#     # model_score = round(max(score) * 100, 2)

#     # return {"result": "Welcome!!!"}
    return {data}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)