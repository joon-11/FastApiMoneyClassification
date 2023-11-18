import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote

import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifieir.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return{"Message" : "기계학습 위조지폐 인식 API"}

@app.post("/predict")
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if(prediction[0]>0.5):
        result = "위조지폐"
    else:
        result = "은행권"

    return {"prediction" : result}

