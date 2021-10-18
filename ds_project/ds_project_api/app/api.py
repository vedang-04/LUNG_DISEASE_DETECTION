import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger

from app import __version__, schemas
from app.config import settings
from image_classification_model import __version__ as model_version
from image_classification_model.config.core import config
from image_classification_model.predict import make_prediction
from image_classification_model.processing.data_manager import create_dataset

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )
    logger.info("health")
    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict() -> Any:
    input_data = create_dataset(folder_name=config.app_config.test_folder, single=False)
    input_df = input_data.copy()
    results = make_prediction(input_data=input_df.replace({np.NaN: None}))
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    return results
