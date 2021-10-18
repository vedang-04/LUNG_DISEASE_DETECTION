from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from image_classification_model.config.core import config
from image_classification_model.processing.data_manager import create_dataset


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return create_dataset(folder_name=config.app_config.test_folder, single=False)


@pytest.fixture()
def client() -> Generator:
    from app.application import app

    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
