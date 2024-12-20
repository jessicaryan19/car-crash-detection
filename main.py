from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load Keras model and preprocessing file
MODEL_PATH = "./models/mobilenetv3.keras"
PKL_PATH = "./models/random_forest_model.pkl"
VGG_19_PATH = "./models/mobilenetv2.keras"
LOGISTIC_PATH = "./models/logistic_regression_model.pkl"

try:
    model = load_model(MODEL_PATH)
    with open(PKL_PATH, "rb") as f:
        preprocessor = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {e}")

# Define request data schema
class InputData(BaseModel):
    input_features: list[float]  # Adjust the type and structure based on your model's expected input

# Define a prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert input to numpy array
        # input_array = np.array(data.input_features).reshape(1, -1)

        # Preprocess input using the .pkl file (e.g., scaling or tokenization)
        # processed_input = preprocessor.transform(input_array)

        

        # Make prediction
        prediction = model.predict(data)

        # Return prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict_random_forest/")
async def predict_random_forest(data: InputData):
    try:
        # Convert input to numpy array
        input_array = np.array(data.input_features).reshape(1, -1)

        # Preprocess input using the .pkl file (e.g., scaling or tokenization)
        processed_input = preprocessor.transform(input_array)

        # Make prediction
        prediction = model.predict(processed_input)

        # Return prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
@app.post("/predict_vgg19/")
async def predict_vgg19(data: InputData):
    try:
        # Convert input to numpy array
        input_array = np.array(data.input_features).reshape(1, -1)

        # Preprocess input using the .pkl file (e.g., scaling or tokenization)
        processed_input = preprocessor.transform(input_array)

        # Make prediction
        prediction = model.predict(processed_input)

        # Return prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
@app.post("/predict_logistic/")
async def predict_logistic(data: InputData):
    try:
        # Convert input to numpy array
        input_array = np.array(data.input_features).reshape(1, -1)

        # Preprocess input using the .pkl file (e.g., scaling or tokenization)
        processed_input = preprocessor.transform(input_array)

        # Make prediction
        prediction = model.predict(processed_input)

        # Return prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")