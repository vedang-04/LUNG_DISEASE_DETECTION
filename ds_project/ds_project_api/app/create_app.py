import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.api import api_router
from app.config import settings
from image_classification_model.predict import make_prediction


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )
    templates = Jinja2Templates(directory=os.getcwd() + r"\app\templates")

    root_router = APIRouter()

    @root_router.get("/")
    def index() -> Any:
        """Basic HTML response."""
        body = (
            "<html>"
            "<body style='padding: 10px;'>"
            "<h1>Welcome to the API</h1>"
            "<div>"
            "Check the docs: <a href='/docs'>here</a>"
            "</div>"
            "</body>"
            "</html>"
        )
        return HTMLResponse(content=body)

    @root_router.get("/imageclassificationform", response_class=HTMLResponse)
    def form_post_get(
        request: Request,
    ):
        result = "Upload the Image"
        return templates.TemplateResponse(
            "ds_project.html", context={"request": request, "result": result}
        )

    @root_router.post("/imageclassificationform")
    async def form_post(request: Request, assignment_file: UploadFile = File(...)):
        try:
            file_to_be_read = (
                os.getcwd() + "\\" + "live_data" + "\\" + str(assignment_file.filename)
            )
            await assignment_file.read()
            input_df = pd.DataFrame([file_to_be_read], columns=["image_name"])
            prediction = make_prediction(input_data=input_df.replace({np.NaN: None}))
            num = prediction.get("predictions")[0]
            if num == 0:
                result = "NO DISEASE"
            elif num == 1:
                result = "BACTERIAL PNEUMONIA"
            else:
                result = "VIRAL PNEUMONIA"
            return templates.TemplateResponse(
                "ds_project.html", context={"request": request, "result": result}
            )
        except Exception:
            result = "Upload the Image"
            return templates.TemplateResponse(
                "ds_project.html", context={"request": request, "result": result}
            )

    app.include_router(api_router, prefix=settings.API_V1_STR)
    app.include_router(root_router)
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_route("/", index)
    app.add_route("/imageclassificationform", form_post_get)
    app.add_route("/imageclassificationform", form_post)
    logger.info("Application instance created")
    return app
