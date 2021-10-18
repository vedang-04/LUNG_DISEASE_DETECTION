import uvicorn
from loguru import logger

from app.config import settings, setup_app_logging
from app.create_app import create_app

setup_app_logging(config=settings)

app = create_app()
logger.info("App Created")

if __name__ == "__main__":
    logger.warning("Running in development mode. Do not run like this in production.")
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
