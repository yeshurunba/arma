#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


def create_arma(data_series, p=3, q=3):
    print("barak")
    series = data_series.values
    print(data_series)
    print(series)
    arma_mod30 = ARIMA(series, order=(p, 0, q)).fit()
    Y = arma_mod30.predict(30, 60)
    return Y


# create_arma(pd.Series([1, 2, 3, 1, 2, 3]))
