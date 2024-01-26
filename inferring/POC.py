import yfinance as yf
import pandas as pd


def get_data():
    ticker = yf.Ticker("LUMI.TA")
    return ticker.history(start="2020-06-02", interval="1d")
    ## return ticker.history(period="max", interval="1wk")


# get_data().to_csv("barak.csv")

df = pd.read_csv("barak.csv")

# print(df)
# print(df["Close"])


import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression


def create_regression(data_series):
    # data_series = pd.read_csv("data_series.csv")  # load data_series set
    X = data_series.index.values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data_series.values.reshape(
        -1, 1
    )  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    new_data = pd.Series((Y - Y_pred).reshape(-1))
    return new_data, linear_regressor


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


def create_arma(data_series, p=7, q=7):
    series = data_series.values
    arma_mod30 = ARIMA(series, order=(p, 0, q)).fit()
    Y = arma_mod30.predict(30, 60)
    return Y


def create_tik(data_series):
    """data_series should be of length 30"""
    series_pd, linear_regressor = create_regression(data_series)
    arma_predict = create_arma(series_pd)
    true_predict = arma_predict + linear_regressor.predict(
        np.array(range(30, 61)).reshape(-1, 1)
    ).reshape(-1)
    if False:  # for simple display
        plt.plot(range(30, 61), true_predict)
        plt.show()
    return pd.Series(true_predict, index=range(30, 61))


# print(series_pd)
# print(arma_predict)
# plt.plot(range(30, 61), arma_predict)
# plt.plot(range(30), series_pd)
# plt.show()


# create_tik(df["Close"][:30])

# tester = 30
# create_tik(df["Close"].iloc[tester:].iloc[:30]).plot()
# plt.plot(range(60), df["Close"].iloc[tester:].iloc[:60])
# plt.show()

df["Close"].plot()

# create_tik(df["Close"].iloc[:30]).plot()
# plt.plot(range(60), df["Close"].iloc[:60])
# plt.show()
