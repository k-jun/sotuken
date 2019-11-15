import boto3
from datetime import date, datetime, timedelta

s3 = boto3.client('s3')
today = date.today()
mintime = datetime.min.time()
now = datetime.combine(today, mintime)
now = now - timedelta(days=1)

for i in range(100):
    now = now - timedelta(days=1)
    fileName = now.strftime("%Y-%m-%d") + '.json'
    s3.download_file('spot-instance-price-tracker',
                     fileName, './s3/' + fileName)
    print(i, fileName)
