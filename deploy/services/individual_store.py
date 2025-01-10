import pandas as pd
import numpy as np
import cloudpickle

with open("models/model_lgbm.pkl", "rb") as f:
    lgbm = cloudpickle.load(f)

with open("models/pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

with open("models/target_encoders.pkl", "rb") as f:
    target_encoder = cloudpickle.load(f)


data = pd.read_parquet("data/deploy_data.parquet")
train_data = pd.read_parquet("data/train_data.parquet")


def transform_predict(data):
    data = data.copy()
    data.loc[:, "Store_id_enc"] = target_encoder.transform(data[["Store_id"]])
    data = pipeline.transform(data)
    return lgbm.predict(data.reshape(1, -1))[0]


def build_future_dataframe(Store_id=1, days=30, train_data=train_data, test_data=test_data, date=first_forecast_date):
    train_data = train_data.copy()
    test_data = test_data.copy()
    last_30_days = train_data[train_data['Date'] >=
                              pd.to_datetime(date) - pd.Timedelta(f'{days} days')]
    last_30_days = last_30_days[last_30_days["Store_id"] == Store_id]

    forecast_data = test_data[test_data["Store_id"] == Store_id]
    forecast_data = forecast_data[forecast_data["Date"] == date]
    assert forecast_data["Date"].iloc[0] == date
    forecast_data[['Sales', 'Orders']] = np.nan, np.nan
    # print(forecast_data.to_dict(orient='records'))

    forecast_data = pd.concat([last_30_days, forecast_data], ignore_index=True)

    # assert forecast_data.shape[0] == last_30_days.shape[0] + 1

    transformed_forecast_data = pipeline.transform(forecast_data)
    forecast = lgbm.predict(transformed_forecast_data)

    forecast_data.iloc[-1, forecast_data.columns.get_loc("Sales")] = forecast
    forecast_data.iloc[-1, forecast_data.columns.get_loc("Orders")] = 0

    return forecast_data


def forecast(Store_id=1, days=31):

    first_forecast_date = pd.to_datetime("2019-06-01")
    updated_forecast = build_future_dataframe(
        Store_id=Store_id, days=30, train_data=train_data, test_data=test_data, date=first_forecast_date)

    for i in range(1, days):
        next_date = first_forecast_date + pd.Timedelta(days=i)
        updated_forecast = build_future_dataframe(
            Store_id=Store_id, days=30, train_data=updated_forecast, test_data=test_data, date=next_date)

    updated_forecast["predicted"] = 'Yes'
    trimmed = train_data[train_data["Store_id"] == Store_id]
    trimmed["predicted"] = 'No'

    return pd.concat([trimmed, updated_forecast], ignore_index=True)
