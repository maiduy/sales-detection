import pickle, json
from flask import Flask, jsonify, request 
import pandas as pd

# Load model
with open("sales-detection/weights/model.pickle", 'rb') as f:
    model_all = pickle.load(f)

scaler = model_all['scaler']
regression_model = model_all['regression_model']

# Parameter for data processing
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

# processing data function

def process_data(df):
    df = pd.DataFrame(df).rename(columns=mapping_fields)
    df["order_short_ym"] = pd.to_datetime(df["order_date"], errors='coerce').dt.strftime('%y%m')
    features = ["order_short_ym", "segment", "region", "sub_category"]
    df['order_short_ym'] = df['order_short_ym'].apply(int)

    def map_sub_category(x):
        return sub_cat_dict[x]
    df["sub_category"] = df["sub_category"].apply(map_sub_category)

    def map_segment(x):
        if x == 'Consumer':
            return 0
        elif x == 'Corporate':
            return 1
        else:
            return 2
    df["segment"] = df["segment"].apply(map_segment)

    def map_region(x):
        if x == 'South':
            return 0
        elif x == 'West':
            return 1
        elif x == 'Central':
            return 2
        else:
            return 3
    df["region"] = df["region"].apply(map_region)

    X = df[features].to_numpy()
    return X


# Flask application init
app = Flask(__name__)

@app.route('/sale-regression', methods=['POST'])
def predict():
    # get data
    req_json = request.get_json()
    df = pd.DataFrame().from_dict(req_json)
    # process data
    X = process_data(df)
    # normalize data
    X_norm = scaler.transform(X)
    # predict
    y_hat = regression_model.predict(X_norm)
    df["predicted_sales"] = y_hat
    # return result
    return df.to_dict(orient='list')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
