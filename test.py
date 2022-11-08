from time import time
import requests
import json 
import pandas as pd


df = pd.read_csv("sales-detection/dataset/train.csv")
req = df[5:7].to_dict(orient='list')
req_json = json.dumps(req)
print(req_json)
print("==========================")
URL = "http://127.0.0.1:5000/sale-regression"

headers = {"content-type": "application/json"}
cur_t = time()
r = requests.post(url=URL, headers=headers, data=req_json)
print(time()-cur_t)
print(r.text)