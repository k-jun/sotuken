import json
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

ZONE = "ap-northeast-1c"
DATA_START_DATE = "2019-07-01"
DATA_END_DATE = "2019-11-01"

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
            if i["zone"] == ZONE:
                df = df.append(i, ignore_index=True)
    current = current - timedelta(days=1)

print(df.head())
df.to_csv("./input/" + ZONE + ".csv", index=False)
