import joblib
import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from pipeline import dis_pip
from processing.data_manager import create_dataset, save_pipeline

from package.image_classification_model import __version__ as _version
from package.image_classification_model.config.core import TRAINED_MODEL_DIR, config


def run_training() -> None:
    train_df = create_dataset(config.app_config.train_folder, single=False)
    val_df = create_dataset(config.app_config.val_folder, single=False)
    data = pd.concat([train_df, val_df]).reset_index(drop=True)
    ohe = OneHotEncoder(variables=["label"])
    data = ohe.fit_transform(data)
    save_enc = TRAINED_MODEL_DIR / f"{config.app_config.saved_enc}{_version}.pkl"
    joblib.dump(ohe, save_enc)
    Y = np.array(data[data.columns[1:]])
    dis_pip.fit(data, Y)
    save_pipeline(pipeline_to_persist=dis_pip)

    from package.image_classification_model.predict import make_prediction

    test_df = create_dataset(folder_name=config.app_config.test_folder, single=False)
    result = make_prediction(input_data=test_df)
    predictions = result.get("predictions")
    print(predictions)
    assert result.get("errors") is None


if __name__ == "__main__":
    run_training()
