import json
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

ZONE = "ap-northeast-1c"
DATA_START_DATE = "2019-07-01"
DATA_END_DATE = "2019-12-01"

now = datetime.strptime(DATA_END_DATE, "%Y-%m-%d")

# define index
dates = pd.date_range(start=DATA_START_DATE, end=DATA_END_DATE)

columns = ["instance_type", "time_stump", "region", "zone", "price"]
df = pd.DataFrame(columns=columns)

# set startDate
current = now
for i in range(len(dates)):
    fileName = "./s3/" + current.strftime("%Y-%m-%d") + ".json"
    with open(fileName) as f:
        data = json.load(f)
        for i in data:
            if i["zone"] == ZONE and i["instance_type"] in ["m3.large", "m5.2xlarge", "m5.large", "m5.xlarge", "r3.xlarge", "r5d.xlarge"]:
                df = df.append(i, ignore_index=True)
    current = current - timedelta(days=1)

print(df.head())
df.to_csv("./input/" + ZONE + "_from_20190701_to_20191201" +
          DATA_END_DATE + ".csv", index=False)
