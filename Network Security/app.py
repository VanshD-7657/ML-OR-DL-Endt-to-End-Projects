from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import os,sys
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from pathlib import Path

from networksecurity.utils.main_utils.utils import load_object
BASE_DIR = Path(__file__).resolve().parent

try:
    client = pymongo.MongoClient(mongo_db_url, tls=True, tlsCAFile=ca, serverSelectionTimeoutMS=20000)
    client.server_info()   # forces connection attempt
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]
    logging.info("Connected to MongoDB Atlas")
except Exception as e:
    logging.warning(f"MongoDB connection failed; continuing without DB: {e}")
    client = None
    database = None
    collection = None
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get('/', tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successfull")
    except Exception as e:
        raise NetworkSecurityException(e,sys)


@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        # read uploaded CSV into dataframe
        df = pd.read_csv(file.file)

        # build absolute paths for model files
        preprocessor_path = BASE_DIR / "final_model" / "preprocessor.pkl"
        model_path = BASE_DIR / "final_model" / "model.pkl"

        preprocessor = load_object(str(preprocessor_path))
        final_model = load_object(str(model_path))

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # predict
        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        # save and render HTML table (optional)
        out_csv = BASE_DIR / "prediction_output" / "output.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__=="__main__":
    app_run(app,host="localhost",port=8000)

