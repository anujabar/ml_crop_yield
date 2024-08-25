from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import numpy as np
import bz2file as bz2

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Crop : int
    Crop_Year : int
    Season : int
    State : int
    Area : float
    Production : int
    Annual_Rainfall :  float
    Fertilizer :  float
    Pesticide :  float
    

# loading the saved model
model = bz2.BZ2File("yield_model_cmp.pbz2", "rb")
model = pickle.load(model)

@app.post('/yield_prediction')
def yield_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    crop = input_dictionary['Crop']
    cy = input_dictionary['Crop_Year']
    season = input_dictionary['Season']
    state = input_dictionary['State']
    area = input_dictionary['Area']
    prod = input_dictionary['Production']
    ar = input_dictionary['Annual_Rainfall']
    fert = input_dictionary['Fertilizer']
    pest = input_dictionary['Pesticide']


    input_list = [crop,cy,season,state,area,prod,ar,fert,pest]
    
    a=np.array(input_list)
    a = a.reshape(1, -1)
    prediction = model.predict(a)
    
    return prediction[0]

