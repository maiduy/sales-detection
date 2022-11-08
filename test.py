from time import time
import requests
import json 
import pandas as pd


df = pd.read_csv("sales-detection/dataset/train.csv")
req = df[:10].to_dict(orient='list')
req_json = json.dumps(req)
print(req_json)

URL = "http://172.17.0.2:5000/sale-regression"

headers = {"content-type": "application/json"}
cur_t = time()
r = requests.post(url=URL, headers=headers, data=req_json)
print(time()-cur_t)
print(r.text)