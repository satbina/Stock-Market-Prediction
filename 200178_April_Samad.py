import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv("sample_input.csv")

    actual_close = np.loadtxt("sample_close.txt")
    pred_close = predict_func(df)

    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close - pred_close))
    

    pred_prev = [df["Close"].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close

    actual_prev = [df["Close"].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr) - np.array(pred_prev)
    actual_dir = np.array(actual_curr) - np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir * actual_dir) > 0) * 100

    print(
        f"Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}"
    )


def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    loaded_model = pickle.load(open("model_0_9_5.sav", "rb"))
    # import keras
    # import tensorflow as tf
    # loaded_model = tf.keras.models.load_model('model_gaurang_sexy.h5')
    df = data
    df = df["Close"]
    df = df.interpolate()
    df1 = df
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(df).reshape(-1, 1))
    X = []
    X.append(df)
    X = np.array(X)
    y = loaded_model.predict(X)
    y = scaler.inverse_transform(y)
    df1.loc[len(df)] = y[0, 0]
    df1 = df1[1 : len(df1)]
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    X1 = []
    X1.append(df1)
    X1 = np.array(X1)
    y1 = loaded_model.predict(X1)
    y1 = scaler.inverse_transform(y1)
    ans = []
    ans.append(y[0, 0])
    ans.append(y1[0, 0])
    return np.array(ans)


if __name__ == "__main__":
    evaluate()
