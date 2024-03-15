from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()


def lode_model(model_path):
    with open(model_path, "rb") as pipeline_file:
        pipe_lr = joblib.load(pipeline_file)
    return pipe_lr

model_path = "/home/pong/work/test/models/ZEN.pkl"
loaded_model = lode_model(model_path)

templates = Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("form.html",{"request": request, "result": None})

@app.post("/",response_class=HTMLResponse)
async def predict_sentiment(request: Request, text: str = Form(...)):
    try :
        predict_sentiment = loaded_model.predict([text])[0]
        if predict_sentiment == "pos":
            result = "positive"
        else:
            result = "negative"

        return templates.TemplateResponse("form.html",{"request": request, "result": result})
    except Exception as e:
        return {"error": "Prediction failed."}