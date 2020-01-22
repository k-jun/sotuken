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
from keras.models import load_model
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


DATA_PATH = "./input/ap-northeast-1c_from_2019-07-01_to_2019-12-01.csv"
DATA_PATH_ACTUAL = "./input/ap-northeast-1c_from_2019-12-01_to_2019-12-10.csv"
# m3.large, m5.2xlarge, m5.large, m5.xlarge, r3.xlarge, r5d.xlarge

TARGET_TYPE = "r3.xlarge"
BATCH_SIZE = 32
EPOCHS = 100
PAST_HISTORY = 10
MULTI_STEP = 10
TRAIN_RATIO = 0.7
RANDOM_STATE = 1221
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

colors = ["red", "royalblue", "violet", "green", "cyan", "orange"]
plt.figure(figsize=(10, 10))
instance_types = ["m3.large", "m5.2xlarge",
                  "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]

for it_index in range(len(instance_types)):
    TARGET_TYPE = instance_types[it_index]

    df = lib.load_data(DATA_PATH, TARGET_TYPE)

    df, mean, std = lib.normalize(df)
    (x_train, y_train), (x_test, y_test), columns = lib.train_test_split_lstm(df["price"].values, df.index,
                                                                              PAST_HISTORY, TRAIN_RATIO)
    # モデルを定義
    model = create_model(x_train.shape[-2:])
    # モデルを学習
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=1, validation_data=(x_test, y_test))

    df_10days = lib.load_data(DATA_PATH_ACTUAL, TARGET_TYPE)
    # モデルの検証
    df_10days, mean, std = lib.normalize(df_10days)
    values = df_10days["price"].values
    dates = df_10days.index

    x_val = []
    y_val = []
    spans = []
    for i in range(PAST_HISTORY, len(values)-MULTI_STEP):
        s = [round((dates[j] - dates[i]).seconds / 3600 + 24 *
                   (dates[j] - dates[i]).days, 2) for j in range(i, i + MULTI_STEP)]
        x_val.append([values[j] for j in range(i - PAST_HISTORY, i)])
        y_val.append([values[j] for j in range(i, i + MULTI_STEP)])
        spans.append(s)
    result = []
    # マルチステップで予測
    for x_item in x_val:
        result_item = []
        for _ in range(MULTI_STEP):
            x_input = np.array([np.reshape(x_item, (PAST_HISTORY, 1))])
            next_item = model.predict(x_input)
            x_item.pop(0)
            x_item.append(float(next_item))
            result_item.append(float(next_item))
        result.append(result_item)

    gosas = []
    times = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            # 誤差
            pred = lib.denormalize([result[i][j]], std, mean)
            true = lib.denormalize([y_val[i][j]], std, mean)
            gosas.append(round((pred[0] - true[0]) / true[0] * 100, 2))
            # 時間
            times.append(spans[i][j])
    plt.scatter(times, gosas, c=colors[it_index], label=TARGET_TYPE)

plt.legend(bbox_to_anchor=(0, 1), loc='upper left',
           borderaxespad=1, fontsize=15)
plt.xlabel('spans(hours)')
plt.ylabel('error(%)')
plt.tight_layout()
plt.xlim([0, 60])
plt.savefig("./output/lstm-10days")
