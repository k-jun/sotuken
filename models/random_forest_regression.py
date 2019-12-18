import tensorflow.keras.backend as K
import random as rn
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.ensemble import RandomForestRegressor

import lib
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# m3.large, m5.2xlarge, m5.large, m5.xlarge, r3.xlarge, r5d.xlarge
TARGET_TYPE = "m5.2xlarge"
DATA_PATH = "./input/ap-northeast-1c-20191213.csv"
MULTI_STEP = 10
BATCH_SIZE = 32
EPOCHS = 50
PAST_HISTORY = 30

RANDOM_STATE = 1222
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
        x_train.append(a)
        y_train.append(dataset[i])

    x_test = []
    y_test = []

    for i in range(split_index, len(dataset)):
        a = []
        for j in range(i-history_size, i):
            a.append(dataset[j])
        x_test.append(a)
        y_test.append(dataset[i])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def create_model(input_shape):
    model = RandomForestRegressor(
        max_depth=5, random_state=RANDOM_STATE, n_estimators=100)
    return model


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

    # 正規化
    df, mean, std = lib.normalize(df)

    print(df.shape)

    x_train, y_train, x_test, y_test = split_dataset(
        df["price"].values, MULTI_STEP, PAST_HISTORY)

    # モデルを定義
    model = create_model(x_train.shape[-2])

    # モデルを学習
    model.fit(x_train, y_train)

    y_train_pred, y_test_pred = predict_model(
        model, x_train, y_train, x_test, y_test, MULTI_STEP)

    # 非正規化
    y_train = lib.denormalize(y_train, std, mean)
    y_test = lib.denormalize(y_test, std, mean)
    y_train_pred = lib.denormalize(y_train_pred, std, mean)
    y_test_pred = lib.denormalize(y_test_pred, std, mean)

    lib.graph(y_train, y_test, y_train_pred, y_test_pred,
              "random_forest_regression", TARGET_TYPE)
    lib.mse(y_train, y_test, y_train_pred, y_test_pred)
