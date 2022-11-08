import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import pickle

# Load data

df = pd.read_csv('sales-detection/dataset/train.csv')

mapping_fields = {
    'Row ID': 'row_id',
    'Order ID': 'order_id',
    'Order Date': 'order_date',
    'Ship Date': 'ship_date',
    'Ship Mode': 'ship_mode',
    'Customer ID': 'customer_id',
    'Customer Name': 'customer_name',
    'Segment': 'segment',
    'Country': 'country',
    'City': 'city',
    'State': 'state',
    'Postal Code': 'postal_code',
    'Region': 'region',
    'Product ID': 'product_id',
    'Category': 'category',
    'Sub-Category': 'sub_category',
    'Product Name': 'product_name',
    'Sales': 'sales'
}
columns = ["row_id", "order_id", "order_date", "ship_date", "ship_mode", "customer_id", "customer_name", "segment",
            "country", "city", "state", "postal_code", "region", "product_id", "category", "sub_category", "product_name", "sales"]

sub_cat_dict = {'Accessories': 0, 'Appliances': 1, 'Art': 2, 'Binders': 3, 'Bookcases': 4, 'Chairs': 5, 'Copiers': 6, 'Envelopes': 7, 'Fasteners': 8, 'Furnishings': 9, 'Labels': 10, 'Machines': 11, 'Paper': 12, 'Phones': 13, 'Storage': 14, 'Supplies': 15, 'Tables': 16}

pdTmp = pd.DataFrame(df).rename(columns=mapping_fields)
pdSales = pdTmp.loc[:,columns]

for column in pdSales.select_dtypes(include=['float64']).columns:
     pdSales[column] = pdSales[column].fillna(0.0).astype('float')

for column in pdSales.select_dtypes(include=['int', 'int64']).columns:
    pdSales[column] = pdSales[column].fillna(0).astype('int')

# for column in ["order_date", "ship_date"]:
#     pdSales[column] = pd.to_datetime(pdSales[column], errors='coerce').dt.strftime('%Y-%m-%d')
pdSales["order_date"] = pd.to_datetime(pdSales["order_date"], errors='coerce').dt.strftime('%Y-%m-%d')
pdSales["order_ym"] = pd.to_datetime(pdSales["order_date"], errors='coerce').dt.strftime('%Y-%m')
pdSales["order_short_ym"] = pd.to_datetime(pdSales["order_date"], errors='coerce').dt.strftime('%y%m')
pdSales["ship_date"] = pd.to_datetime(pdSales["ship_date"], errors='coerce').dt.strftime('%Y-%m-%d')
pdSales["ship_ym"] = pd.to_datetime(pdSales["ship_date"], errors='coerce').dt.strftime('%Y-%m')
pdSales['order_quarter'] = pd.PeriodIndex(pd.to_datetime(pdSales["order_date"], errors='coerce'), freq='Q')

str_columns = ["ship_mode", "customer_id", "customer_name", "segment",
            "country", "city", "state", "region", "product_id", "category", "sub_category", "product_name"]

for column in str_columns:
     pdSales[column] = pdSales[column].fillna('').astype('str')


# Processing data

train_df = pdSales.copy()

features = ["order_short_ym", "segment", "region", "sub_category"]
target = ["sales"]


train_df['order_short_ym'] = train_df['order_short_ym'].apply(int)

def map_sub_category(x):
    return sub_cat_dict[x]

train_df["sub_category"] = train_df["sub_category"].apply(map_sub_category)


def map_segment(x):
    if x == 'Consumer':
        return 0
    elif x == 'Corporate':
        return 1
    else:
        return 2
train_df["segment"] = train_df["segment"].apply(map_segment)


def map_region(x):
    if x == 'South':
        return 0
    elif x == 'West':
        return 1
    elif x == 'Central':
        return 2
    else:
        return 3
train_df["region"] = train_df["region"].apply(map_region)

X = train_df[features].to_numpy()
y = train_df[target].to_numpy()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
X_norm = scaler.fit_transform(X)


# Init Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train Model
model.fit(X_norm, y)
print(model.coef_)

# Save Model
final_model = {
    "scaler": scaler,
    "regression_model": model
}

with open("sales-detection/weights/model.pickle", "wb") as f:
    pickle.dump(final_model, f)