
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import random as rn
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from datetime import datetime
import statsmodels.api as sm

DATA_PATH = "./input/ap-northeast-1c.csv"
TARGET_TYPE = "c5d.2xlarge"
TRAIN_SPLIT = 400

RANDOM_STATE = 1221
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_data(path, i_type):
    df = pd.read_csv(DATA_PATH)
    df = df[df["instance_type"] == i_type]
    s_date = pd.Series(
        [pd.to_datetime(date) for date in df["time_stump"]]
    )
    df = df.set_index(s_date)
    df = df.drop(["zone", "region", "instance_type", "time_stump"], axis=1)
    df = df.iloc[::-1]
    return df


def data_graph(df):
    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(
        df["time_stump"],
        df["price"],
        label=TARGET_TYPE
    )
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
               borderaxespad=1, fontsize=10)
    plt.show()
    return


def data_hist(df):
    plt.figure(figsize=(16, 9), dpi=50)
    plt.hist(df["price"], bins=100, density=True, label=TARGET_TYPE)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
               borderaxespad=1, fontsize=10)
    plt.show()
    return


def normalize(df):
    mean = df["price"].mean()
    std = df["price"].std()

    # print("平均: ", mean)
    # print("最大値: ", df["price"].max())
    # print("最小値: ", df["price"].min())
    # print("個数: ", len(df["price"].values))
    # print("標準偏差: ", df["price"].std())
    # print("分散: ", df["price"].var(ddof=0))

    if np.isnan(std) == 0.0:
        df["price"] = df["price"] - mean
    else:
        df["price"] = (df["price"] - mean) / std

    return df


def data_decide_pq(df):
    info_criteria = sm.tsa.stattools.arma_order_select_ic(
        df.values, ic=['aic', 'bic']
    )

    print("aic_min_order: ", info_criteria.aic_min_order)
    print("bic_min_order: ", info_criteria.bic_min_order)


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


def result_graph(arma_pred, df):
    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(arma_pred.index, arma_pred.values, color="g", label="predict")
    plt.plot([i for i in range(df.shape[0])], df["price"].values,
             color="b", label="actual_train")
    # plt.plot(x_index[len(y_train):], y_test, color="r", label="actual_test")
    plt.show()


#################################################
#                  mainの処理                    #
#################################################


df = load_data(DATA_PATH, TARGET_TYPE)
df = normalize(df)

data_decide_pq(df)

# df_train = df.iloc[:TRAIN_SPLIT, :]
# df_test = df.iloc[TRAIN_SPLIT:, :]
# print(df_train.shape)
# print(df_test.shape)

arma_fit = sm.tsa.ARMA(df, (4, 2)).fit()
arma_pred = arma_fit.predict(start=0, end=df.shape[0] + 10)
result_graph(arma_pred, df)
