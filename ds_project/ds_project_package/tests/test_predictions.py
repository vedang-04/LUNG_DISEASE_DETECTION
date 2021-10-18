import numpy as np
import pandas as pd

from image_classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):

    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    print(predictions)
    assert result.get("errors") is None
