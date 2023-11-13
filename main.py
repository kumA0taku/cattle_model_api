from fastapi import FastAPI, Query, HTTPException
# from fastapi.responses import JSONResponse
from typing import List
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from json import dumps
from uvicorn import run
import os
import numpy as np
from io import StringIO

app = FastAPI()

new_model = tf.keras.models.load_model('c2_N35_cell2.h5', compile=False)
new_model.compile()
new_model.summary()

# Define the class names
class_names = ["stationary", "sleep"]

# Define a function to preprocess the input data
def preprocess_data(input_data):
    # Parse the CSV string into a DataFrame
    df = pd.read_csv(StringIO(input_data))

    # Replace behavior labels
    df = df.replace('Stationary-Or-Eating', 0)
    df = df.replace('sleeping', 1)

    # Extract features and target
    y = df['behavior'].to_numpy().reshape((-1, 1))
    X = df.drop(columns=['behavior']).to_numpy()

    # Create dataset suitable for LSTM
    dataX, dataY = create_dataset(X, y, 35)

    return dataX

# Function to create dataset suitable for LSTM
def create_dataset(dataset_input, dataset_result, n_steps):
    dataX, dataY = [], []
    for i in range(len(dataset_input) - n_steps + 1):
        a = dataset_input[i:(i + n_steps), :]
        dataX.append(a)
        dataY.append(dataset_result[i + n_steps - 1, :])
    return np.array(dataX), np.array(dataY)

@app.get("/")
async def root():
    return {"message":"Version 0.0.1"}

# Endpoint to make predictions
@app.post("/predict/")
# async def predict(data: str = Query(..., description="Comma-separated data in the format acc_x,acc_y,acc_z,gy_x,gy_y,gy_z,behavior")):
async def predict(data: List[List[List[float]]]):

    # Make predictions
    predictions = new_model.predict(data)

    # # Get prediction classes
    prediction_classes = np.argmax(predictions, axis=1)

    # # Map prediction classes to class names
    predicted_labels = [class_names[pred] for pred in prediction_classes]

    return {"predicted_labels": predicted_labels}
    # return {data}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)