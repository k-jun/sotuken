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

# models
import lightgbm
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

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


instance_types = ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]
# instance_types = ["m3.large"]
for i in range(len(instance_types)):
    # for i in ["m3.large"]:
    TARGET_TYPE = instance_types[i]
    print("=" * 10, TARGET_TYPE, "=" * 10)

    df = lib.load_data(DATA_PATH, TARGET_TYPE)
    df, mean, std = lib.normalize(df)

    (x_train_lstm, y_train_lstm), (x_test_lstm, y_test_lstm), columns = lib.train_test_split_lstm(df["price"].values, df.index,
                                                                              PAST_HISTORY, TRAIN_RATIO)
    (x_train, y_train), (x_test, y_test), columns = lib.train_test_split(df["price"].values, df.index,
                                                                              PAST_HISTORY, TRAIN_RATIO)
    # モデルを定義
    lstm = create_model(x_train_lstm.shape[-2:])
    rfr = RandomForestRegressor(max_depth=5, random_state=RANDOM_STATE, n_estimators=100)
    xgb = xgboost.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    lgbm = lightgbm.LGBMRegressor(n_estimators=100, random_state=RANDOM_STATE)
    svm = SVR()
    # モデルを学習
    lstm.fit(x_train_lstm, y_train_lstm, batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=1, validation_data=(x_test_lstm, y_test_lstm))
    rfr.fit(x_train, y_train)
    xgb.fit(x_train, y_train)
    lgbm.fit(x_train, y_train)
    svm.fit(x_train, y_train)

    # モデルで予測
    y_pred_lstm = lstm.predict(x_test_lstm)
    y_pred_rfr = rfr.predict(x_test)
    y_pred_xgb = xgb.predict(x_test)
    y_pred_lgbm = lgbm.predict(x_test)
    y_pred_svm = svm.predict(x_test)

    # 非正規化
    y_pred_lstm = lib.denormalize(y_pred_lstm, std, mean)
    y_pred_rfr = lib.denormalize(y_pred_rfr, std, mean)
    y_pred_xgb = lib.denormalize(y_pred_xgb, std, mean)
    y_pred_lgbm = lib.denormalize(y_pred_lgbm, std, mean)
    y_pred_svm = lib.denormalize(y_pred_svm, std, mean)
    y_test = lib.denormalize(y_test, std, mean)


    plt.figure(figsize=(16, 9), dpi=50)
    
    # x_index =  - len(y_pred_rfr)
    x_index = df.index[-len(y_test):]

    plt.plot(x_index, y_test, color="black", label="actual")
    plt.plot(x_index, y_pred_lstm, color="g", label="lstm predict")
    plt.plot(x_index, y_pred_rfr, color="r", label="rfr predict")
    plt.plot(x_index, y_pred_xgb, color="b", label="xgb predict")
    plt.plot(x_index, y_pred_lgbm, color="purple", label="lgbm predict")
    plt.plot(x_index, y_pred_svm, color="orange", label="svm predict")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=15)
    plt.savefig("./output/all-" + TARGET_TYPE + ".png")

for i in result:
    print(i)
