import json
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

ZONE = "ap-northeast-1d"
DATA_START_DATE = "2019-07-01"
DATA_END_DATE = "2019-12-10"


def columns(dates):
    a = []
    for d in dates:
        fileName = "./s3/" + str(d)[:10] + ".json"
        with open(fileName) as f:
            data = json.load(f)
            for i in data:
                if i["zone"] == ZONE and not i["instance_type"] in a:
                    a.append(i["instance_type"])
    return a


def zones(dates):
    a = []
    for d in dates:
        fileName = "./s3/" + str(d)[:10] + ".json"
        with open(fileName) as f:
            data = json.load(f)
            for i in data:
                if not i["zone"] in a:
                    a.append(i["zone"])
    return a


def data(dates, df):
    for d in dates:
        fileName = "./s3/" + str(d)[:10] + ".json"
        with open(fileName) as f:
            data = json.load(f)
            for i in data:
                if i["zone"] == ZONE:
                    df.at[str(d)[:10], i["instance_type"], ] = i["price"]

    return df


def main():
    dates = pd.date_range(start=DATA_START_DATE, end=DATA_END_DATE)
    c = columns(dates.values)
    z = zones(dates)
    print(c)
    print(z)
    # df = pd.DataFrame(columns=c)
    # df = data(dates.values, df)
    #
    # df.to_csv("./preview/" + ZONE + ".csv")


main()
