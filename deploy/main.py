from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load the pre-trained sklearn model
model = joblib.load("/path/to/your/model.pkl")


class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    prediction: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
