from sklearn.pipeline import Pipeline

from image_classification_model.config.core import config
from image_classification_model.create_model import kc
from image_classification_model.processing.features import CreateDataset

dis_pip = Pipeline(
    steps=[
        ("cd", CreateDataset(target_size=config.model_config.target_size)),
        ("estimator", kc),
    ]
)
