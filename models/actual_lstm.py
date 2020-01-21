# ref: https://www.tensorflow.org/guide/keras/rnn
# ref: https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score, mean_squared_error

import lib


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)
    # return K.maximum(K.sign(y_true - y_pred), 0.01) * tf.keras.losses.mean_absolute_error(y_true, y_pred)


def create_model(input_shape):
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, input_shape=input_shape),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Activation("linear")
    ])

    lstm_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=custom_loss,
    )

    lstm_model.summary()
    return lstm_model

def predict_model(model, x_train, y_train, x_test, y_test, multi_step):
    y_train_pred = []
    y_test_pred = []

    for i in x_train:
        y_train_pred.append(model.predict(np.array([i]))[0])

    # multi step predict
    prev_input = list(x_test[0])
    for i in range(multi_step):
        latest_input = model.predict(np.array([prev_input]))[0]
        y_test_pred.append(latest_input)
        prev_input.append(latest_input)
        prev_input.pop(0)
    return y_train_pred, y_test_pred

DATA_PATH = "./input/ap-northeast-1c_from_20190701_to_201912012019-12-01.csv"
# m3.large, m5.2xlarge, m5.large, m5.xlarge, r3.xlarge, r5d.xlarge

TARGET_TYPE = "r5d.xlarge"
MULTI_STEP = 10
BATCH_SIZE = 32
EPOCHS = 100
PAST_HISTORY = 10
TRAIN_RATIO = 0.7

RANDOM_STATE = 1221
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


result = []


colors = ["red", "royalblue", "violet", "green", "cyan", "orange"]
# instance_types = ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]
instance_types = ["m3.large"]

for i in range(len(instance_types)):
    subfig = fig.add_subplot(2, 3, i+1)
    # for i in ["m3.large"]:
    TARGET_TYPE = instance_types[i]
    print("=" * 10, TARGET_TYPE, "=" * 10)

    df = lib.load_data(DATA_PATH, TARGET_TYPE)
    df, mean, std = lib.normalize(df)

    (x_train, y_train), (x_test, y_test), columns = lib.train_test_split_lstm(df["price"].values, df.index,
                                                                              PAST_HISTORY, TRAIN_RATIO)
    # モデルを定義
    model = create_model(x_train.shape[-2:])
    # モデルを学習
    hist = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=1, validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)

    a = {}
    a["r2_score"] = r2_score(y_pred, y_test)
    a["rmse"] = np.sqrt(mean_squared_error(y_pred, y_test))
    result.append(a)
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

for i in result:
    print(i)