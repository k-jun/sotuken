import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


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


# def data_hist(df):
#     plt.figure(figsize=(16, 9), dpi=50)
#     plt.hist(df["price"], bins=100, density=True, label=TARGET_TYPE)
#     plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
#                borderaxespad=1, fontsize=10)
#     plt.show()
#     return


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


def train_test_split(values, dates, history_size, train_ratio):
    columns = ["10days-ago", "9days-ago", "8days-ago", "7days-ago", "6days-ago",
               "5days-ago", "4days-ago", "3days-ago", "2days-ago", "1days-ago"
               #    "month", "dayofmonth", "hour", "dayofweek"
               ]

    x_train = pd.DataFrame(columns=columns)
    y_train = []
    x_test = pd.DataFrame(columns=columns)
    y_test = []

    train_index = int(len(values) * train_ratio)

    for i in range(history_size, train_index):
        a = {}
        for j in range(1, history_size+1):
            a[str(j) + "days-ago"] = values[i - j]
        # a["month"] = dates[i].month
        # a["dayofmonth"] = dates[i].day
        # a["hour"] = dates[i].hour
        # a["dayofweek"] = dates[i].dayofweek
        x_train = x_train.append(a, ignore_index=True)
        y_train.append(values[i])

    for i in range(train_index + history_size, len(values)):
        a = {}
        for j in range(1, history_size+1):
            a[str(j) + "days-ago"] = values[i - j]
        # a["month"] = dates[i].month
        # a["dayofmonth"] = dates[i].day
        # a["hour"] = dates[i].hour
        # a["dayofweek"] = dates[i].dayofweek
        x_test = x_test.append(a, ignore_index=True)
        y_test.append(values[i])

    return (x_train.values, np.array(y_train)), (x_test.values, np.array(y_test)), columns


def train_test_split_lstm(values, dates, history_size, train_ratio):
    (x_train, y_train), (x_test, y_test), columns = train_test_split(
        values, dates, history_size, train_ratio)

    x_train_new = np.array(list(
        map(lambda x: np.reshape(x, (len(columns), 1)), x_train)))
    x_test_new = np.array(
        list(map(lambda x: np.reshape(x, (len(columns), 1)), x_test)))

    return (x_train_new, y_train), (x_test_new, y_test), columns


def graph(y_train, y_test, y_train_pred, y_test_pred, model_type, target_type, train_date, test_date):
    x_index = train_date + test_date
    plt.figure(figsize=(16, 9), dpi=50)
    plt.plot(x_index, y_train_pred + y_test_pred, color="g", label="predict")
    plt.plot(x_index[:len(y_train)], list(y_train), color="b", label="train")
    plt.plot(x_index[len(y_train):], list(y_test), color="r", label="test")

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=1, fontsize=15)
    plt.xlabel("Date")
    plt.ylabel("Doll per Hour")
    plt.title("model: " + model_type + " instance_type: " + target_type)
    plt.savefig("./output/" + model_type + "/" +
                target_type + ".png")
