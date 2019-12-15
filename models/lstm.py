# ref: https://www.tensorflow.org/guide/keras/rnn
# ref: https://www.tensorflow.org/tutorials/structured_data/time_series

from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import random as rn
import tensorflow.keras.backend as K

import lib

DATA_PATH = "./input/ap-northeast-1c-20191213.csv"
# m3.large, m5.2xlarge, m5.large, m5.xlarge, r3.xlarge, r5d.xlarge

TARGET_TYPE = "r5d.xlarge"
MULTI_STEP = 10
BATCH_SIZE = 32
EPOCHS = 100
PAST_HISTORY = 30

RANDOM_STATE = 1221
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def split_dataset(dataset, multi_step, history_size):
    split_index = int(len(dataset) - multi_step)
    x_train = []
    y_train = []

    for i in range(history_size, split_index):
        a = []
        for j in range(i-history_size, i):
            a.append(dataset[j])
        x_train.append(np.reshape(a, (history_size, 1)))
        y_train.append(dataset[i])

    x_test = []
    y_test = []

    for i in range(split_index, len(dataset)):
        a = []
        for j in range(i-history_size, i):
            a.append(dataset[j])
        x_test.append(np.reshape(a, (history_size, 1)))
        y_test.append(dataset[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def custom_loss(y_true, y_pred):
    # return tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return K.maximum(K.sign(y_true - y_pred), 0.5) * tf.keras.losses.mean_absolute_error(y_true, y_pred)


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


#################################################
#                  mainの処理                    #
#################################################


for i in ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]:
    TARGET_TYPE = i

    df = lib.load_data(DATA_PATH, TARGET_TYPE)
    df, mean, std = lib.normalize(df)

    # data_hist(df)
    # lib.data_graph(df, TARGET_TYPE)

    x_train, y_train, x_test, y_test = split_dataset(
        df["price"].values, MULTI_STEP, PAST_HISTORY)

    # # モデルを定義
    model = create_model(x_train.shape[-2:])

    # # モデルを学習
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
            verbose=1, validation_data=(x_test, y_test))

    y_train_pred, y_test_pred = predict_model(
        model, x_train, y_train, x_test, y_test, MULTI_STEP)

    y_train = lib.denormalize(y_train, std, mean)
    y_test = lib.denormalize(y_test, std, mean)
    y_train_pred = lib.denormalize(y_train_pred, std, mean)
    y_test_pred = lib.denormalize(y_test_pred, std, mean)

    lib.graph(y_train, y_test, y_train_pred, y_test_pred, "lstm", TARGET_TYPE)
    lib.mse(y_train, y_test, y_train_pred, y_test_pred)
