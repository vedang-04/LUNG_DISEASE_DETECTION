import joblib
import pytest

from image_classification_model.config.core import TRAINED_MODEL_DIR, config
from image_classification_model.processing.data_manager import create_dataset


@pytest.fixture()
def sample_input_data():
    test_df = create_dataset(folder_name=config.app_config.test_folder)
    return test_df


@pytest.fixture(scope="session")
def pipeline_inputs():
    train_df = create_dataset(folder_name=config.app_config.train_folder)
    val_df = create_dataset(folder_name=config.app_config.val_folder)
    data = pd.concat([train_df, val_df]).reset_index(drop=True)
    path = TRAINED_MODEL_DIR / config.app_config.saved_enc
    ohe = joblib.load(path)
    data = ohe.transform(data)
    Y = np.array(data[data.columns[1:]])
    return data, Y


@pytest.fixture()
def raw_input_data():
    return create_dataset(folder_name=config.app_config.train_folder)
