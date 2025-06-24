from fastapi import FastAPI
import pickle
import pandas
from pydantic import BaseModel
import os

class PredictionRequest(BaseModel):
    x: list[float]

app = FastAPI()
model_path = "./app/ml_model.pkl"

model = pickle.load(open(model_path, "rb"))
features = column = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]

@app.get("/")
def read_root():
    return {"message":"ML model is ready"}

@app.post("/predict")
def prediction(request_: PredictionRequest):
    x = request_.x
    y = model.predict(pandas.DataFrame([{f:v for f,v in zip(features, x)}]))
    return {"prediction":int(y[0])}