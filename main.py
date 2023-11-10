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



@app.get("/")
async def root():
    return {"message":"Version 0.0.1"}

@app.post("/predict/")
async def predict(data: str =""):
    return {"data":type(data)}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)