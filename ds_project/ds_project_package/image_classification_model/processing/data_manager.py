import typing as t
from glob import glob

import joblib
import pandas as pd
from keras.models import load_model
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from image_classification_model import __version__ as _version
from image_classification_model.config.core import (
    DATASET_DIR,
    TRAINED_MODEL_DIR,
    config,
)


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    save_file_name_model = f"{config.app_config.pipeline_save_file}{_version}.h5"
    save_path_model = TRAINED_MODEL_DIR / save_file_name_model
    model_save_file_name = f"{config.app_config.pipeline_save_file_model}{_version}.pkl"
    model_save_path = TRAINED_MODEL_DIR / model_save_file_name
    print(pipeline_to_persist)
    cnn_model = pipeline_to_persist.named_steps["estimator"]
    print(cnn_model)
    try:
        joblib.dump(cnn_model.classes_, model_save_path)
    except Exception as e:
        print(e)
    cnn_model.model.save(save_path_model)
    print(pipeline_to_persist.named_steps["estimator"])
    pipeline_to_persist.named_steps["estimator"].model = None
    pipeline_to_persist.named_steps["estimator"].classes_ = None
    print(pipeline_to_persist.named_steps["cd"])
    pipeline_to_dump = Pipeline(
        steps=[("cd", pipeline_to_persist.named_steps["cd"]), ("estimator", None)]
    )
    joblib.dump(pipeline_to_dump, save_path)
    remove_old_pipelines(
        files_to_keep=[save_file_name, save_file_name_model, model_save_file_name]
    )
    del pipeline_to_persist


def save_pipeline_new(*, pipeline_to_persist: Pipeline) -> None:
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file_new}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    save_file_name_model = f"{config.app_config.pipeline_save_file_new}{_version}.h5"
    save_path_model = TRAINED_MODEL_DIR / save_file_name_model
    model_save_file_name = (
        f"{config.app_config.pipeline_save_file_model_new}{_version}.pkl"
    )
    model_save_path = TRAINED_MODEL_DIR / model_save_file_name
    cnn_model = pipeline_to_persist.named_steps["estimator"]
    cnn_model.model.save(save_path_model)
    pipeline_to_persist.named_steps["estimator"] = None
    remove_old_pipelines(
        files_to_keep=[save_file_name, save_file_name_model, model_save_file_name]
    )
    joblib.dump(pipeline_to_persist, save_path)
    joblib.dump(cnn_model.classes_, model_save_path)


def load_pipeline(*, pipeline_to_load: Pipeline) -> Pipeline:
    load_file_name = TRAINED_MODEL_DIR / pipeline_to_load
    save_file_name_model = f"{config.app_config.pipeline_save_file}{_version}.h5"
    save_path_model = TRAINED_MODEL_DIR / save_file_name_model
    model_save_file_name = f"{config.app_config.pipeline_save_file_model}{_version}.pkl"
    model_save_path = TRAINED_MODEL_DIR / model_save_file_name

    trained_pipeline = joblib.load(filename=load_file_name)

    def build_model():
        return load_model(save_path_model)

    rlr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.01,
        patience=5,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )
    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
    kc = KerasClassifier(
        build_fn=build_model,
        batch_size=config.model_config.batch_size,
        epochs=config.model_config.epochs,
        verbose=1,
        validation_split=config.model_config.test_size,
        callbacks=[es, rlr],
    )
    kc.classes_ = joblib.load(model_save_path)
    kc.model = build_model()
    trained_pipeline_to_load = Pipeline(
        steps=[("cd", trained_pipeline.named_steps["cd"]), ("estimator", kc)]
    )
    return trained_pipeline_to_load


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def create_dataset(folder_name) -> pd.DataFrame:
    df_n = pd.DataFrame(
        glob(
            str(DATASET_DIR)
            + "/"
            + config.app_config.folder
            + "/"
            + folder_name
            + "/"
            + config.app_config.subfolder
            + "/*"
        ),
        columns=["image_name"],
    )
    try:
        df_n["label"] = df_n.image_name.apply(
            lambda x: x.split("/",)[3].split(
                "\\"
            )[0]
        )
    except:
        df_n["label"] = df_n.image_name.apply(
            lambda x: x.split("/",)[10]
        )
    l1 = []
    l2 = []
    for x in glob(
        str(DATASET_DIR)
        + "/"
        + config.app_config.folder
        + "/"
        + folder_name
        + "/"
        + config.app_config.subfolder_d
        + "/*"
    ):
        if "bacteria" in x:
            l1.append(x)
        else:
            l2.append(x)
    df_b = pd.DataFrame(l1, columns=["image_name"])
    try:
        df_b["label"] = df_b.image_name.apply(
            lambda b: config.app_config.subfolder_d
            + "_"
            + b.split(
                "/",
            )[3]
            .split("\\")[1]
            .split("_")[1]
            .upper()
       )
    except:
        df_b["label"] = df_b.image_name.apply(
            lambda b: b.split(
                "/",
            )[10]
            + "_"
            + b.split(
                "/",
            )[11]
            .split("_")[1]
            .upper()
       )
    df_v = pd.DataFrame(l2, columns=["image_name"])
    #print(df_n.image_name.iloc[0].split('/',))
    #print(df_b.image_name.iloc[0].split('/',))
    #print(df_v.image_name.iloc[0].split('/',))
    try:
        df_v["label"] = df_v.image_name.apply(
            lambda v: config.app_config.subfolder_d
            + "_"
            + v.split(
                "/",
            )[3]
            .split("\\")[1]
            .split("_")[1]
            .upper()
        )
    except:
        df_v["label"] = df_v.image_name.apply(
            lambda v: v.split(
                "/",
            )[10]
            + "_"
            + v.split(
                "/",
            )[11]
            .split("_")[1]
            .upper()
       )
    df_i = pd.concat([df_n, df_b]).reset_index(drop=True)
    df = pd.concat([df_i, df_v]).reset_index(drop=True)
    return df
