import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(tickers, start_date, end_date):
    data_dict = {}

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not df.empty and "Close" in df.columns:
            data_dict[ticker] = df["Close"]

    if not data_dict:
        raise ValueError("No data downloaded.")

    data = pd.concat(data_dict.values(), axis=1)
    data.columns = list(data_dict.keys())
    data.dropna(inplace=True)

    return data


def normalize_data(dataframe):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataframe)
    scaled_df = pd.DataFrame(scaled, index=dataframe.index, columns=dataframe.columns)
    return scaled_df, scaler


def create_sliding_windows(dataframe, window_size=64, target_col=0):
    X = []
    y = []
    dates = []

    values = dataframe.values

    for i in range(window_size, len(values) - 1):
        X.append(values[i - window_size:i])
        y.append(values[i + 1, target_col])
        dates.append(dataframe.index[i + 1])

    return np.array(X), np.array(y), dates


def inverse_transform_target(normalized_values, scaler, n_features, target_index):
    temp = np.zeros((len(normalized_values), n_features))
    temp[:, target_index] = normalized_values
    inverse = scaler.inverse_transform(temp)
    return inverse[:, target_index]