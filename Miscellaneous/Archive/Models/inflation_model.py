import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from Simulation.Data.cpi_dk import inflation
from statsmodels.tsa.arima.model import ARIMA

"""
====================================================================================================
Plotting the inflation data
====================================================================================================
"""
plt.figure(figsize=(10, 6))
plt.plot(inflation, label='Inflation')
plt.xlabel('Date')
plt.ylabel('Inflation Rate')
plt.title('Inflation Time Series')
plt.legend()
plt.show()


"""
====================================================================================================
Assessing Stationarity
====================================================================================================
"""
#ADF test
result=adfuller(inflation,maxlag=10,regression='c',autolag='AIC')
print("ADF Statistic: {:.4f}".format(result[0]))
print("p-value: {:.4f}".format(result[1]))



"""
====================================================================================================
Assessing Autocorrelation (ACF) and Partial Autocorrelation (PACF)
====================================================================================================
"""
#Plot ACF and PACF for the original series (if stationary) or the differenced series
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plot_acf(inflation, lags=20, ax=ax[0])
ax[0].set_title('ACF of Differenced Series')

plot_pacf(inflation, lags=20, ax=ax[1])
ax[1].set_title('PACF of Differenced Series')

plt.tight_layout()
plt.show()

#The plots suggest an AR(1) model




"""
====================================================================================================
Fit ARIMA model
====================================================================================================
"""
# Here we assume d=1 from the differencing test and based on the ACF/PACF we choose p=1, q=1
model = ARIMA(inflation, order=(1, 0, 0))
model_fit = model.fit()

# Print a summary of the model
print(model_fit.summary())


"""
====================================================================================================
Diagnostic Checks
====================================================================================================
"""
residuals = model_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.show()

# Plot the ACF of residuals
plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.show()

# You might also want to look at a histogram and Q-Q plot
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of Residuals')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()


"""
====================================================================================================
Forecasting with the Fitted Model
====================================================================================================
"""
# Forecasting the next 12 periods (e.g., months or weeks depending on your data frequency)
forecast_steps = 48
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_conf_int = forecast_result.conf_int()

# Print the forecast
print(forecast_mean)

# Plot the observed series and the forecast
plt.figure(figsize=(12, 6))
plt.plot(inflation, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_conf_int.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Inflation')
plt.title('Inflation Forecast')
plt.legend()
plt.show()
