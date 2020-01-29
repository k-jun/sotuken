
import random as rn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import r2_score, mean_squared_error


import lib

DATA_PATH = "./input/ap-northeast-1c_from_2019-07-01_to_2019-12-01.csv"
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
fig = plt.figure(figsize=(16, 9))
colors = ["red", "royalblue", "violet", "green", "cyan", "orange"]
instance_types = ["m3.large", "m5.2xlarge",
                  "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]
for i in range(len(instance_types)):
    # for i in ["m3.large"]:
    TARGET_TYPE = instance_types[i]
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
    y_pred = lib.denormalize(y_pred, std, mean)
    y_test = lib.denormalize(y_test, std, mean)

    a = {}
    a["r2_score"] = 1 - ((1 - r2_score(y_pred, y_test)) *
                         (len(y_pred) - 1) / (len(y_pred) - 10 - 1))
    a["rmse"] = np.sqrt(mean_squared_error(y_pred, y_test))
    result.append(a)

    subfig = fig.add_subplot(2, 3, i+1)
    subfig.scatter(y_test, y_pred, c="black", label=TARGET_TYPE)
    subfig.set_xlabel('y_test')
    subfig.set_ylabel('y_pred')
    subfig.plot([-2, 4], [-2, 4])
    subfig.legend(bbox_to_anchor=(1, 0), loc='lower right',
                  borderaxespad=1, fontsize=15)

# plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
#            borderaxespad=1, fontsize=15)
# plt.plot([-2, 4], [-2, 4])
# plt.xlabel('y_test')
# plt.ylabel('y_pred')
fig.tight_layout()
fig.savefig("./output/xgb.png")

for i in result:
    print(i)
