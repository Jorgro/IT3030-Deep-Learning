from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import numpy as np
import data_settings as settings


def filter_column_based_on_quantile(df, q, columns):
    for col in columns:
        q_low = df[col].quantile(q)
        q_hi = df[col].quantile(1 - q)
        df = df[(df[col] < q_hi) & (df[col] > q_low)]
    return df


def add_date_time_features(df, one_hot_encode=False):
    # Convert start time to a Datatime-object which is much easier to work with
    df["start_time"] = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S")
    df["time_of_day"] = df["start_time"].dt.hour
    df["time_of_week"] = df["start_time"].dt.dayofweek
    df["time_of_year"] = df["start_time"].dt.month - 1
    df["time_of_hour"] = df["start_time"].dt.minute // 5

    if one_hot_encode:
        # This is just to create atleast one of each possible unique value so get_dummies will work even
        # though the dataframe might not have all possible unique values
        df_date_features = pd.DataFrame()
        df_date_features["time_of_day"] = pd.Series(list(range(0, 24)))
        df_date_features["time_of_week"] = pd.Series(list(range(0, 7)))
        df_date_features["time_of_year"] = pd.Series(list(range(0, 12)))
        df_date_features["time_of_hour"] = pd.Series(list(range(0, 12)))
        df_date_features.fillna(0)

        # One hot encode data-time columns
        temp = pd.get_dummies(
            pd.concat([df, df_date_features], keys=[0, 1]),
            columns=["time_of_hour", "time_of_day", "time_of_week", "time_of_year"],
        )

        # Selecting data from multi index
        df = temp.xs(0)
        return df

    # Else Cos encoding:
    df["time_of_day_cos"] = np.cos(df["time_of_day"] * 2 * np.pi / 24)
    df["time_of_week_cos"] = np.cos(df["time_of_week"] * 2 * np.pi / 7)
    df["time_of_year_cos"] = np.cos(df["time_of_year"] * 2 * np.pi / 12)
    df["time_of_hour_cos"] = np.cos(df["time_of_hour"] * 2 * np.pi / 12)
    return df


def normalize_columns(df, columns: List[str], scaler: MinMaxScaler = None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler


def add_structural_imbalance(df):
    # Find indices we will use as x-axis for interpolation
    start_indice = 11 - df["time_of_hour"][0]
    hour_start_indices = np.arange(start_indice, df.shape[0], 12)

    # Flow is oppposite because of dataset
    sum_prod_flow = -df["flow"].values + df["total"].values
    # Interpolate
    tck = interpolate.splrep(hour_start_indices, sum_prod_flow[hour_start_indices], s=1)
    interpolation = interpolate.splev(np.array(df.index), tck, der=0)

    df["interpolation"] = interpolation
    df["structural_imbalance"] = sum_prod_flow - interpolation
    df["sum"] = sum_prod_flow
    return df


def add_lag_features(df):
    # Previous y lag feature
    df["y_prev"] = df["y"].shift(1)

    # Add power imbalance from 24 hours ago
    df["y_prev_24h"] = df["y"].shift(24 * 60 // 5)

    # Mean y yesterday
    df = pd.merge_asof(
        df,
        df.resample("D", on="start_time")["y"].mean().shift(1),
        right_index=True,
        left_on="start_time",
    )
    df = df.rename(columns={"y_x": "y", "y_y": "y_yesterday"})
    return df


def preprocess_dataframe(df, scaler=None):
    df = add_date_time_features(df, settings.ONE_HOT_ENCODE_TIME)
    df = add_structural_imbalance(df)
    if settings.AVOID_STRUCTURAL_IMBALANCE:
        df["y"] = df["y"] - df["structural_imbalance"]
    df = filter_column_based_on_quantile(df, 0.001, settings.COLUMNS_TO_CLAMP)
    df, scaler = normalize_columns(df, settings.COLUMNS_TO_NORMALIZE, scaler=scaler)
    df = add_lag_features(df)
    df["y_prev"] = df["y_prev"] + np.random.normal(
        0, settings.GAUSSIAN_NOISE, df["y_prev"].shape
    )
    df = df.drop(columns=settings.COLUMNS_TO_DROP)
    return df, scaler

