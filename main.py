
# Convert the 'date' column to datetime format and sort the data

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Re-importing the data due to code execution environment reset
file_path = 'C:/fi/301210_041223_TLV.csv'
ta35_data = pd.read_csv(file_path)

# Convert the 'date' column to datetime format and sort the data
ta35_data['date'] = pd.to_datetime(ta35_data['date'], dayfirst=True)
ta35_data.sort_values('date', inplace=True)

# Calculate daily returns (percentage change)
ta35_data['daily_return'] = ta35_data['locking'].pct_change() * 100

# Remove the first row as its return will be NaN
ta35_data = ta35_data.iloc[1:]

# Generate a time variable (number of days since start)
ta35_data['time'] = np.arange(1, len(ta35_data) + 1)

# Generate a time squared variable
ta35_data['time_squared'] = ta35_data['time'] ** 2

# Prepare the regression model
X = ta35_data[['time', 'time_squared']]
X = sm.add_constant(X)  # adding a constant
y = ta35_data['daily_return']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Generate predicted values from the model
ta35_data['predicted_return'] = model.predict(X)

# Plotting Actual vs. Predicted Values
plt.figure(figsize=(15, 8))
plt.plot(ta35_data['date'], ta35_data['daily_return'], label='Actual Returns', alpha=0.7)
plt.plot(ta35_data['date'], ta35_data['predicted_return'], label='Predicted Returns', alpha=0.7)
plt.title('Actual vs Predicted Daily Returns of TA-35 Index')
plt.xlabel('Date')
plt.ylabel('Daily Returns (%)')
plt.legend()
plt.show()

# Plotting Residuals
residuals = ta35_data['daily_return'] - ta35_data['predicted_return']
plt.figure(figsize=(15, 8))
plt.plot(ta35_data['date'], residuals, label='Residuals', color='red', alpha=0.7)
plt.title('Residuals of the Regression Model Over Time')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')  # Line at y=0 for reference
plt.legend()
plt.show()



# Assuming ta35_data is your DataFrame containing the TA-35 index data
ta35_data['year'] = ta35_data['date'].dt.year
annual_data = ta35_data.groupby('year').agg({'locking': ['first', 'last']})
annual_data['annual_return'] = (annual_data['locking']['last'] - annual_data['locking']['first']) / annual_data['locking']['first'] * 100
annual_data.reset_index(inplace=True)

# Generating time and time squared variables for the annual data
annual_data['time'] = np.arange(1, len(annual_data) + 1)
annual_data['time_squared'] = annual_data['time'] ** 2

# Preparing the regression model for annual data
X_annual = annual_data[['time', 'time_squared']]
X_annual = sm.add_constant(X_annual)  # adding a constant
y_annual = annual_data['annual_return']

# Fitting the regression model for annual data
model_annual = sm.OLS(y_annual, X_annual).fit()

# Predicting annual returns using the fitted model
annual_data['predicted_return'] = model_annual.predict(X_annual)

# Plotting Actual vs Predicted Annual Returns
plt.figure(figsize=(15, 8))
plt.plot(annual_data['year'], annual_data['annual_return'], label='Actual Annual Returns', alpha=0.7)
plt.plot(annual_data['year'], annual_data['predicted_return'], label='Predicted Annual Returns', alpha=0.7)
plt.title('Actual vs Predicted Annual Returns of TA-35 Index')
plt.xlabel('Year')
plt.ylabel('Annual Returns (%)')
plt.legend()
plt.show()

# Plotting Residuals for annual data
residuals_annual = annual_data['annual_return'] - annual_data['predicted_return']
plt.figure(figsize=(15, 8))
plt.plot(annual_data['year'], residuals_annual, label='Residuals', color='red', alpha=0.7)
plt.title('Residuals of the Annual Regression Model')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')  # Line at y=0 for reference
plt.legend()
plt.show()


# Calculate monthly returns
ta35_data['year_month'] = ta35_data['date'].dt.to_period('M')
monthly_data = ta35_data.groupby('year_month').agg({'locking': ['first', 'last']})
monthly_data['monthly_return'] = (monthly_data['locking']['last'] - monthly_data['locking']['first']) / monthly_data['locking']['first'] * 100
monthly_data.reset_index(inplace=True)

# Generate time variables for monthly data
monthly_data['time'] = np.arange(1, len(monthly_data) + 1)
monthly_data['time_squared'] = monthly_data['time'] ** 2

# Prepare the regression model for monthly data
X_monthly = monthly_data[['time', 'time_squared']]
X_monthly = sm.add_constant(X_monthly)
y_monthly = monthly_data['monthly_return']

# Fit the regression model for monthly data
model_monthly = sm.OLS(y_monthly, X_monthly).fit()

# Predicting monthly returns using the fitted model
monthly_data['predicted_return'] = model_monthly.predict(X_monthly)

import matplotlib.pyplot as plt
import numpy as np  # Ensure numpy is imported

# Assuming 'monthly_data' has columns 'year_month', 'monthly_return', and 'predicted_return'

# Determine the interval for displaying x-axis labels
label_interval = 6  # For example, show a label every 6 months

# Adjust the figure size as needed
plt.figure(figsize=(15, 8))

# Plot Actual vs. Predicted Monthly Returns
plt.plot(monthly_data['year_month'].astype(str), monthly_data['monthly_return'], label='Actual Monthly Returns', alpha=0.7)
plt.plot(monthly_data['year_month'].astype(str), monthly_data['predicted_return'], label='Predicted Monthly Returns', alpha=0.7)
plt.title('Actual vs Predicted Monthly Returns of TA-35 Index')
plt.xlabel('Year-Month')
plt.ylabel('Monthly Returns (%)')

# Set x-axis labels at the specified interval
x_ticks = monthly_data['year_month'].astype(str)[::label_interval]
plt.xticks(np.arange(0, len(monthly_data), step=label_interval), x_ticks, rotation=90)
plt.legend()
plt.show()

# Assuming 'monthly_data' contains the actual monthly returns and the predicted returns from your model
residuals_monthly = monthly_data['monthly_return'] - monthly_data['predicted_return']

# Plot Residuals for Monthly Data
# Plotting residuals for monthly data
plt.figure(figsize=(15, 8))
plt.plot(monthly_data['year_month'].astype(str), residuals_monthly, label='Residuals', color='red', alpha=0.7)
plt.title('Residuals of the Monthly Regression Model')
plt.xlabel('Year-Month')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(np.arange(0, len(monthly_data), step=label_interval), x_ticks, rotation=90)
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
import matplotlib.pyplot as plt
import warnings

# Re-importing the data due to code execution environment reset
#file_path = '/mnt/data/301210_041223_TLV.csv'
ta35_data = pd.read_csv(file_path)

# Convert the 'date' column to datetime format and sort the data
ta35_data['date'] = pd.to_datetime(ta35_data['date'], dayfirst=True)
ta35_data.sort_values('date', inplace=True)

# Calculate monthly returns
ta35_data['year_month'] = ta35_data['date'].dt.to_period('M')
monthly_data = ta35_data.groupby('year_month').agg({'locking': ['first', 'last']})
monthly_data['monthly_return'] = (monthly_data['locking']['last'] - monthly_data['locking']['first']) / monthly_data['locking']['first'] * 100
monthly_data.reset_index(inplace=True)

# Check for stationarity
result = adfuller(monthly_data['monthly_return'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Finding the optimal order of the ARMA model
warnings.filterwarnings("ignore")  # Ignore convergence warnings for this step
try:
    # The arma_order_select_ic function automatically fits different ARMA models and selects the best
    res = arma_order_select_ic(monthly_data['monthly_return'].dropna(), ic=['aic', 'bic'], trend='n', max_ar=4, max_ma=4)
    print("AIC", res.aic_min_order)
    print("BIC", res.bic_min_order)
except Exception as e:
    print("An error occurred during ARMA order selection:", e)

# Optionally, plot ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(monthly_data['monthly_return'].dropna(), lags=20, ax=ax[0])
plot_pacf(monthly_data['monthly_return'].dropna(), lags=20, ax=ax[1])
plt.show()

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings

# Assuming that the 'ta35_data' DataFrame is already loaded and contains the daily return data

# Calculate daily returns if not already done
ta35_data['daily_return'] = ta35_data['locking'].pct_change() * 100
ta35_data = ta35_data.dropna()  # Drop NaN values that arise from pct_change()

# Check for stationarity of daily returns
adf_result = adfuller(ta35_data['daily_return'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])

# Finding the optimal order of the ARMA model for daily data
warnings.filterwarnings("ignore")  # Ignore convergence warnings for this step
try:
    # Use 'n' for no trend in arma_order_select_ic
    res_daily = arma_order_select_ic(ta35_data['daily_return'], ic=['aic', 'bic'], trend='n', max_ar=4, max_ma=4)
    print("AIC", res_daily.aic_min_order)
    print("BIC", res_daily.bic_min_order)
except Exception as e:
    print("An error occurred during ARMA order selection for daily data:", e)

# Optionally, plot ACF and PACF for daily data
fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(ta35_data['daily_return'], lags=20, ax=ax[0])
plot_pacf(ta35_data['daily_return'], lags=20, ax=ax[1])
plt.show()


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Fit the ARMA model (using AIC suggestion)
p, q = 3, 3  # ARMA(3,3)
model_arma = ARIMA(ta35_data['daily_return'], order=(p, 0, q))
results_arma = model_arma.fit()

# Generate ARMA predictions
ta35_data['arma_pred'] = results_arma.predict(start=0, end=len(ta35_data)-1)

# Plotting Actual vs. ARMA Predicted Returns
plt.figure(figsize=(15, 8))
plt.plot(ta35_data['date'], ta35_data['daily_return'], label='Actual Daily Returns', alpha=0.7)
plt.plot(ta35_data['date'], ta35_data['arma_pred'], label='ARMA Predicted Returns', alpha=0.7)
plt.title('Actual vs ARMA Predicted Daily Returns of TA-35 Index')
plt.xlabel('Date')
plt.ylabel('Daily Returns (%)')
plt.legend()
plt.show()

# Assume ta35_data contains 'daily_return' and 'arma_pred'

# Buy and Hold Strategy
initial_investment = 100
cumulative_return_bh = (ta35_data['daily_return'] / 100 + 1).cumprod()
final_value_bh = initial_investment * cumulative_return_bh.iloc[-1]

# ARMA Model-Based Strategy
# Buy (1) when predicted return is positive, Sell (-1) or do nothing (0) when negative
arma_signals = np.where(ta35_data['arma_pred'] > 0, 1, -1)
daily_return_arma = ta35_data['daily_return'] * arma_signals
cumulative_return_arma = (daily_return_arma / 100 + 1).cumprod()
final_value_arma = initial_investment * cumulative_return_arma.iloc[-1]

print("Final value with Buy and Hold: NIS", final_value_bh)
print("Final value with ARMA Model-Based Strategy: NIS", final_value_arma)
# ... [previous code] ...

# ARMA Model-Based Strategy
# Buy (1) when predicted return is positive, Sell (-1) or do nothing (0) when negative

from scipy.stats import ttest_ind

arma_signals = np.where(ta35_data['arma_pred'] > 0, 1, -1)
daily_return_arma = ta35_data['daily_return'] * arma_signals
cumulative_return_arma = (daily_return_arma / 100 + 1).cumprod()
final_value_arma = initial_investment * cumulative_return_arma.iloc[-1]

print("Final value with Buy and Hold: NIS", final_value_bh)
print("Final value with ARMA Model-Based Strategy: NIS", final_value_arma)

# Conducting a t-test to compare the returns of the two strategies
# Null hypothesis: There is no significant difference between the strategies
t_stat, p_value = ttest_ind(cumulative_return_bh, cumulative_return_arma)

print("T-statistic:", t_stat) #I fixed the issue 
print("P-value:", p_value)

# Preparing the data for analysis
# Converting 'date' to datetime format and calculating 'daily_return'
ta35_data['date'] = pd.to_datetime(ta35_data['date'], dayfirst=True)
ta35_data['daily_return'] = ta35_data['locking'].pct_change() * 100  # percentage change for returns

# Dropping the first row as it will have NaN value for 'daily_return'
ta35_data = ta35_data.dropna()

# Implementing the ARMA model
from statsmodels.tsa.arima.model import ARIMA

p, q = 3, 3  # ARMA(3,3)
model_arma = ARIMA(ta35_data['daily_return'], order=(p, 0, q))
results_arma = model_arma.fit()

# Generate ARMA predictions
ta35_data['arma_pred'] = results_arma.predict(start=0, end=len(ta35_data)-1)

# Buy and Hold Strategy
initial_investment = 100
cumulative_return_bh = (ta35_data['daily_return'] / 100 + 1).cumprod()

# ARMA Model-Based Strategy
arma_signals = np.where(ta35_data['arma_pred'] > 0, 1, -1)
daily_return_arma = ta35_data['daily_return'] * arma_signals
cumulative_return_arma = (daily_return_arma / 100 + 1).cumprod()

# Plotting the development of profits
plt.figure(figsize=(15, 8))
plt.plot(ta35_data['date'], initial_investment * cumulative_return_bh, label='Buy and Hold Strategy', alpha=0.7)
plt.plot(ta35_data['date'], initial_investment * cumulative_return_arma, label='ARMA Model-Based Strategy', alpha=0.7)
plt.title('Comparison of Investment Strategies: Buy and Hold vs ARMA Model-Based')
plt.xlabel('Date')
plt.ylabel('Investment Value (NIS)')
plt.legend()
plt.grid(True)
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Assuming 'cumulative_return_bh' and 'cumulative_return_arma' are already calculated
# Convert cumulative returns to percentage differences
percentage_difference_bh = (cumulative_return_bh - 1) * 100
percentage_difference_arma = (cumulative_return_arma - 1) * 100

# Plotting Gaussian Distribution for Buy and Hold Strategy
plt.figure(figsize=(15, 8))
sns.histplot(percentage_difference_bh, kde=True, stat="density", linewidth=0)
mu_bh, std_bh = norm.fit(percentage_difference_bh)
xmin_bh, xmax_bh = plt.xlim()
x_bh = np.linspace(xmin_bh, xmax_bh, 100)
p_bh = norm.pdf(x_bh, mu_bh, std_bh)
plt.plot(x_bh, p_bh, 'k', linewidth=2)
plt.title(f'Buy and Hold Strategy: mu = {mu_bh:.2f}, std = {std_bh:.2f}')
plt.xlabel('Percentage Difference in Cumulative Returns')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Plotting Gaussian Distribution for ARMA Model-Based Strategy
plt.figure(figsize=(15, 8))
sns.histplot(percentage_difference_arma, kde=True, stat="density", linewidth=0)
mu_arma, std_arma = norm.fit(percentage_difference_arma)
xmin_arma, xmax_arma = plt.xlim()
x_arma = np.linspace(xmin_arma, xmax_arma, 100)
p_arma = norm.pdf(x_arma, mu_arma, std_arma)
plt.plot(x_arma, p_arma, 'k', linewidth=2)
plt.title(f'ARMA Model-Based Strategy: mu = {mu_arma:.2f}, std = {std_arma:.2f}')
plt.xlabel('Percentage Difference in Cumulative Returns')
plt.ylabel('Density')
plt.grid(True)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load your data
file_path = 'C:/fi/301210_041223_TLV.csv'
ta35_data = pd.read_csv(file_path)

# Calculate 'daily_return' and handle non-finite values
ta35_data['daily_return'] = ta35_data['locking'].pct_change() * 100
ta35_data['daily_return'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN
ta35_data.dropna(subset=['daily_return'], inplace=True)  # Drop rows with NaN values

# [Include steps to calculate daily_return_arma]

# Plotting Gaussian Distribution for Buy and Hold Strategy's Daily Returns
plt.figure(figsize=(15, 8))
sns.histplot(ta35_data['daily_return'], kde=True, stat="density", linewidth=0)
mu_bh_daily, std_bh_daily = norm.fit(ta35_data['daily_return'])
xmin_bh_daily, xmax_bh_daily = plt.xlim()
x_bh_daily = np.linspace(xmin_bh_daily, xmax_bh_daily, 100)
p_bh_daily = norm.pdf(x_bh_daily, mu_bh_daily, std_bh_daily)
plt.plot(x_bh_daily, p_bh_daily, 'k', linewidth=2)
plt.title(f'Buy and Hold Strategy Daily Returns: mu = {mu_bh_daily:.2f}, std = {std_bh_daily:.2f}')
plt.xlabel('Daily Return (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# [Include the plotting code for the ARMA Model-Based Strategy's Daily Returns]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

# Load your data
file_path = 'C:/fi/301210_041223_TLV.csv'
ta35_data = pd.read_csv(file_path)

# Convert the 'date' column to datetime format and calculate daily returns
ta35_data['date'] = pd.to_datetime(ta35_data['date'], dayfirst=True)
ta35_data['daily_return'] = ta35_data['locking'].pct_change() * 100
ta35_data.dropna(subset=['daily_return'], inplace=True)  # Drop rows with NaN values

# Fit the ARMA model (assuming p, d, q are already determined)
p, d, q = 3, 0, 3  # Example values for ARIMA(p,d,q)
model_arma = ARIMA(ta35_data['daily_return'], order=(p, d, q))
results_arma = model_arma.fit()

# Store the ARMA predictions
ta35_data['arma_pred'] = results_arma.predict(start=0, end=len(ta35_data)-1)

# Calculate ARMA Model-Based Strategy's daily returns
arma_signals = np.where(ta35_data['arma_pred'] > 0, 1, -1)
daily_return_arma = ta35_data['daily_return'] * arma_signals

# Plotting Gaussian Distribution for ARMA Model-Based Strategy's Daily Returns
plt.figure(figsize=(15, 8))
sns.histplot(daily_return_arma, kde=True, stat="density", linewidth=0)
mu_arma_daily, std_arma_daily = norm.fit(daily_return_arma)
xmin_arma_daily, xmax_arma_daily = plt.xlim()
x_arma_daily = np.linspace(xmin_arma_daily, xmax_arma_daily, 100)
p_arma_daily = norm.pdf(x_arma_daily, mu_arma_daily, std_arma_daily)
plt.plot(x_arma_daily, p_arma_daily, 'k', linewidth=2)
title = 'ARMA Model-Based Strategy Daily Returns'
plt.title(f'{title}: mu = {mu_arma_daily:.2f}, std = {std_arma_daily:.2f}')
plt.xlabel('Daily Return (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()
