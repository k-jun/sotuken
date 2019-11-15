import tensorflow.keras.backend as K
import random as rn
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.ensemble import RandomForestRegressor


# ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# m3.large
# m5.xlarge
# r3.xlarge
# r5d.xlarge
# m5.large

DATA_PATH = "./input/ap-northeast-1c.csv"
TARGET_TYPE = "m5.large"
TRAIN_SPLIT = 400
BATCH_SIZE = 32
EPOCHS = 50
PAST_HISTORY = 10
FUTURE_TARGET = 0

RANDOM_STATE = 1221
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_data(path, i_type):
    df = pd.read_csv(DATA_PATH)
    df = df[df["instance_type"] == i_type]
    df = df.drop(["zone", "region", "instance_type"], axis=1)
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


def split_dataset(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        sdata = []
        for j in range(i-history_size, i):
            sdata.append(dataset[j])
        data.append(sdata)
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


def custom_loss(y_true, y_pred):
    # return tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return K.maximum(K.sign(y_true - y_pred), 0.5) * tf.keras.losses.mean_absolute_error(y_true, y_pred)


def create_model(input_shape):
    model = RandomForestRegressor(
        max_depth=5, random_state=RANDOM_STATE, n_estimators=100)

    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(),
    #     loss=custom_loss,
    # )
    return model


def result_graph(model, x_train, y_train, x_test, y_test):
    y_pred = []
    for i in range(len(x_train)):
        y_pred.append(model.predict(np.array([x_train[i]]))[0])

    for i in range(len(x_test)):
        y_pred.append(model.predict(np.array([x_test[i]]))[0])

    x_index = range(len(y_pred))

    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(x_index, y_pred, color="g", label="predict")
    plt.plot(x_index[:len(y_train)], list(y_train),
             color="b", label="actual_train")
    plt.plot(x_index[len(y_train):], y_test, color="r", label="actual_test")
    plt.show()


#################################################
#                  mainの処理                    #
#################################################


df = load_data(DATA_PATH, TARGET_TYPE)
df = normalize(df)

# data_hist(df)
data_graph(df)

# x_train, y_train = split_dataset(
#     df["price"].values, 0, TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET)
# x_test, y_test = split_dataset(
#     df["price"].values, TRAIN_SPLIT, None, PAST_HISTORY, FUTURE_TARGET)

# print("x_train: ", x_train.shape, "y_train: ", y_train.shape)
# print("x_test: ", x_test.shape, "y_test: ", y_test.shape)

# # モデルを定義
# model = create_model(x_train.shape[-2:])

# # モデルを学習
# model.fit(x_train, y_train)

# result_graph(model, x_train, y_train, x_test, y_test)
