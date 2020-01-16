
import random as rn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import r2_score, mean_squared_error


import lib

DATA_PATH = "./input/ap-northeast-1c_from_20190701_to_201912012019-12-01.csv"
# ref: https://github.com/dmlc/xgboost
TARGET_TYPE = "m3.large"
MULTI_STEP = 0
BATCH_SIZE = 32
EPOCHS = 50
PAST_HISTORY = 10
TRAIN_RATIO = 0.7

RANDOM_STATE = 1000
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)

result = []
for i in ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]:
    # for i in ["m3.large"]:
    TARGET_TYPE = i
    df = lib.load_data(DATA_PATH, TARGET_TYPE)

    # 正規化
    df, mean, std = lib.normalize(df)

    (x_train, y_train), (x_test, y_test), columns = lib.train_test_split(df["price"].values, df.index,
                                                                         PAST_HISTORY, TRAIN_RATIO)

    # モデルを定義
    model = xgboost.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    # モデルを学習
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("r2_score", r2_score(y_pred, y_test))
    print("rmse: ", np.sqrt(mean_squared_error(y_pred, y_test)))

    # 非正規化
    # y_train = lib.denormalize(y_train, std, mean)
    # y_test = lib.denormalize(y_test, std, mean)
    # y_train_pred = lib.denormalize(y_train_pred, std, mean)
    # y_test_pred = lib.denormalize(y_test_pred, std, mean)

    fi = model.feature_importances_
    for i in range(len(fi)):
        print(columns[i], fi[i])


for i in result:
    print(i)
