import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path, i_type):
    df = pd.read_csv(data_path)
    df = df[df["instance_type"] == i_type]
    s_date = pd.Series(
        [pd.to_datetime(date) for date in df["time_stump"]]
    )
    df = df.set_index(s_date)
    df = df.drop(["zone", "region", "instance_type", "time_stump"], axis=1)
    df = df.iloc[::-1]
    return df


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
    if np.isnan(std):
        df["price"] = df["price"] - mean
    else:
        df["price"] = [(i - mean) / std for i in df["price"]]
    return df, mean, std


def denormalize(dataset, std, mean):
    return [i * std + mean for i in dataset]


def save_graph(df, target_type, path):
    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(
        df.index.values,
        df["price"].values,
        label=target_type
    )
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
               borderaxespad=1, fontsize=10)
    # plt.show()
    plt.savefig(path)
    return


def graph(y_train, y_test, y_train_pred, y_test_pred, model_type, target_type):
    x_index = range(len(y_train) + len(y_test))

    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(x_index, y_train_pred + y_test_pred, color="g")
    plt.plot(x_index[:len(y_train)], list(y_train), color="b")
    plt.plot(x_index[len(y_train):], list(y_test), color="r")
    plt.savefig("./output/" + model_type + "/" +
                target_type + ".png")


def mse(y_train, y_test, y_train_pred, y_test_pred):
    error_train = [y_train[i] - y_train_pred[i] for i in range(len(y_train))]
    error_test = [y_test[i] - y_test_pred[i] for i in range(len(y_test))]
    MSE_train = np.array([e ** 2 for e in error_train]).mean()
    MSE_test = np.array([e ** 2 for e in error_test]).mean()
    print("MSE_train", MSE_train, "MSE_test", MSE_test)
