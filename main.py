import matplotlib.pyplot as plt

import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models
from sklearn.model_selection import train_test_split

from FeaturesData import FeaturesDataset
import pandas as pd

if __name__ == "__main__":
    SQ_BASE = "database.sqlite"
    QUERY = "SELECT * FROM  `sounds+akg`"
    PATH = 'features_dataset.csv'

    EPOCHS = 50
    BATCH_SIZE = 1000

    training_dataset = FeaturesDataset(SQ_BASE, QUERY)
    training_dataset.sq_connect()
    _, y = training_dataset.data_prep()
    training_dataset.get_features()
    training_data = training_dataset.unpack_features()


    X_train, X_valid, y_train, y_valid = train_test_split(
        training_data, y, train_size=0.7, random_state=42
    )

    model = models.Sequential(
        [
            layers.Input(shape=X_train.shape[1:]),
            layers.Normalization(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(len(list(y.unique())), activation="softmax", name="instruments"),
        ]
    )

    model.compile(
        optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        # callbacks=callbacks.EarlyStopping(verbose=1, patience=2),
    )

    metrics = history.history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    # plt.savefig('2023_07_01_accuracy.png')
    plt.show()