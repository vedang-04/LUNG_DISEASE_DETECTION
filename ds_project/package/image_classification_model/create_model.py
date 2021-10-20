from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from package.image_classification_model.config.core import config


def model_cnn():
    vgg = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(config.model_config.target_size, config.model_config.target_size)
        + (3,),
    )
    for layer in vgg.layers:
        layer.trainable = False
    x = Flatten()(vgg.output)
    prediction = Dense(units=config.model_config.target_classes, activation="softmax")(
        x
    )
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )
    return model


rlr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
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
    build_fn=model_cnn,
    batch_size=config.model_config.batch_size,
    epochs=config.model_config.epochs,
    verbose=1,
    validation_split=config.model_config.test_size,
    callbacks=[es, rlr],
)
