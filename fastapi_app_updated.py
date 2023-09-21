
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pickle

app = FastAPI()

# Serve static files like CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
with open('logistic_model_reduced.pkl', 'rb') as file:
    model = pickle.load('/Users/matheusrech/Documents/logistic_model_reduced_files/logistic_model_reduced.pkl')

@app.get("/")
def read_root():
    return FileResponse("/Users/matheusrech/Documents/logistic_model_reduced_files/final_interactive_prototype_with_updated_title.html")

@app.post("/predict")
async def predict(features: list):
    try:
        prediction = model.predict([features])[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
