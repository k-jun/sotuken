
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
TARGET_TYPE = "m5.2xlarge"
RANDOM_STATE = 1221
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
for date in ["2019-12-01", "2019-12-02", "2019-12-03", "2019-12-04", "2019-12-05", "2019-12-06", "2019-12-07", "2019-12-08", "2019-12-09", "2019-12-10"]:
    for instance_type in ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]:
        try:
            TARGET_TYPE = instance_type
            DATA_PATH = "./input/ap-northeast-1c-20191220-" + date + ".csv"

            df = lib.load_data(DATA_PATH, TARGET_TYPE)
            df, mean, std = lib.normalize(df)

            print("df.shape", df.shape)

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
            result.append((TARGET_TYPE, DATA_PATH, max(y_test) / max(y_test_pred)))
        except:
            continue

for i in result:
    print(i)
