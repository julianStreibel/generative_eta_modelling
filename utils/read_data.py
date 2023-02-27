import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np

PATH_TO_TRAIN_FOLDER = Path("eta_data/train")
PATH_TO_DATA_FILE = Path("eta_data/data.csv")


def read_data_from_txt_files():
    dfs = list()
    for file_name in tqdm(os.listdir(PATH_TO_TRAIN_FOLDER)):
        file_df = pd.read_csv(
            PATH_TO_TRAIN_FOLDER / file_name, sep="\s{2,}", index_col=0
        ).T
        file_df = file_df.drop(file_df.columns[-1], axis=1)
        dfs.append(file_df)
    return pd.concat(dfs, axis=0)


def read_data():
    return pd.read_csv(PATH_TO_DATA_FILE)


def preprocessing(df):
    numerical_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Time_taken (min)",
    ]
    categorical_columns = [
        "Weather conditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
    ]
    time_columns = [
        "Time_Orderd",
        "Time_Order_picked",
    ]
    date_columns = [
        "Order_Date",
    ]

    numericals = df[numerical_columns]
    categoricals = df[categorical_columns].astype("category")
    categoricals = categoricals.apply(lambda x: x.cat.codes)
    times = df[time_columns].apply(fixing_time_strings)
    times = times.apply(lambda x: "1900-01-01-" + x)
    times = times.apply(lambda x: pd.to_datetime(x))
    times = times.apply(lambda x: x.apply(lambda x: save_toordinal(x)))
    dates = df[date_columns].apply(lambda x: pd.to_datetime(x))
    dates = dates.apply(lambda x: x.apply(lambda x: save_toordinal(x)))

    return pd.concat([numericals, categoricals, times, dates], axis=1)


def save_toordinal(x):
    if isinstance(x, pd.Timestamp):
        return x.toordinal()
    return x


def fixing_time_strings(series):
    """
    some time strings have 60 in the minute splot fixing it increasing hour by one and setting minutes to 0
    some time strings have 24 as hour setting it to 0
    """

    def fix(stri):
        if not isinstance(stri, str):
            return stri
        hour, minute = [int(c) for c in stri.split(":")]
        if minute == 60:
            minute = 0
            hour += 1
        hour = hour % 24
        return f"{hour}:{minute}"

    return series.apply(fix)


if __name__ == "__main__":
    df = read_data_from_txt_files()
    df = preprocessing(df)
    df.to_csv(PATH_TO_DATA_FILE, index=False)
