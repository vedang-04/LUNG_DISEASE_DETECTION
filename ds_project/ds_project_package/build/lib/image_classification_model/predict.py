import typing as t

import pandas as pd

from image_classification_model import __version__ as _version
from image_classification_model.config.core import config
from image_classification_model.processing.data_manager import load_pipeline

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
dis_pipe = load_pipeline(pipeline_to_load=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    predictions = dis_pipe.predict(X=data)
    results = {
        "predictions": [pred for pred in predictions],
        "version": _version,
        "errors": None,
    }
    return results
