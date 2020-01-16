
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
import statsmodels.api as sm

import lib

DATA_PATH = "./input/ap-northeast-1c-20191213.csv"
# m3.large, m5.2xlarge, m5.large, m5.xlarge, r3.xlarge, r5d.xlarge
TARGET_TYPE = "r3.xlarge"
RANDOM_STATE = 1222
PREDICT_STEP = 10
np.random.seed(RANDOM_STATE)
rn.seed(RANDOM_STATE)


def data_decide_pq(df):
    info_criteria = sm.tsa.stattools.arma_order_select_ic(
        df.values, ic=['aic']
    )

    return info_criteria.aic_min_order

#################################################
#                  mainの処理                    #
#################################################


result = []
for i in ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]:
    TARGET_TYPE = i

    df = lib.load_data(DATA_PATH, TARGET_TYPE)
    df, mean, std = lib.normalize(df)

    split_point = int(df.shape[0] - PREDICT_STEP)
    df_train = df.iloc[:split_point, :]
    df_test = df.iloc[split_point:, :]

    aic = data_decide_pq(df_train)
    try:
        arma_fit = sm.tsa.ARMA(df_train, aic).fit()
    except:
        arma_fit = sm.tsa.ARMA(df_train, (aic[0], 0)).fit()

    arma_pred = arma_fit.predict(start=0, end=df.shape[0]-1)

    y_train = lib.denormalize(df_train["price"].values, std, mean)
    y_test = lib.denormalize(df_test["price"].values, std, mean)
    y_train_pred = lib.denormalize(arma_pred[:len(y_train)], std, mean)
    y_test_pred = lib.denormalize(arma_pred[len(y_train):], std, mean)

    train_date = []
    for i in df.index[:len(df_train)]:
        train_date.append(i)

    lib.graph(y_train, [], y_train_pred, [],
              "arma", TARGET_TYPE, train_date, [])

    MSE_train, MSE_test = lib.rmse(y_train, y_test, y_train_pred, y_test_pred)
    print("y_test", y_test)
    print("y_test", y_test_pred)
    R2_train, R2_test = lib.r2(y_train, y_test, y_train_pred, y_test_pred)
    result.append({
        "MSE_train": MSE_train,
        "MSE_test": MSE_test,
        "R2_train": R2_train,
        "R2_test": R2_test
    })

for i in result:
    print(i)

# [0.0682, 0.0682, 0.0682, 0.0682, 0.0682, 0.0682, 0.0682, 0.0682, 0.0682, 0.0682]
# [0.06814527233740428, 0.06809433411571837, 0.06804713374440584, 0.06800361691185508, 0.06796372672463097, 0.06792740384562891, 0.06789458663100778, 0.0678652112657827, 0.06783921189796357, 0.06781652077113066]
