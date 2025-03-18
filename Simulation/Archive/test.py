import pandas as pd
import matplotlib.pyplot as plt

"""
Data Preparation and Visualisation
"""

# Load the data
data=pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Archive/Data/cpi_dk.xlsx", parse_dates=[0], index_col=0)
inflation = data['Forbrugerprisindeks']

# Plot the inflation series
plt.figure(figsize=(10, 4))
plt.plot(inflation.index, inflation, label='Monthly Inflation')
plt.title('Monthly Inflation Rate (1980 - Present)')
plt.xlabel('Date')
plt.ylabel('Inflation (%)')
plt.legend()
plt.show()


"""
Implementing a Markov-Switching Model
"""
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Fit a Markov-switching model with 2 regimes
# Here we allow the constant (trend) and variance to switch between regimes.
ms_model = MarkovRegression(inflation, k_regimes=2, trend='c', switching_variance=True)
ms_result = ms_model.fit()
print(ms_result.summary())

# Plot the smoothed probability of being in regime 0
smoothed_probs = ms_result.smoothed_marginal_probabilities[0]

plt.figure(figsize=(10, 4))
plt.plot(smoothed_probs.index, smoothed_probs, label='Probability of Regime 0')
plt.title('Smoothed Regime Probabilities')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.legend()
plt.show()


"""
Forecasting Future Inflation
"""
# Forecast the next 12 months
predicted = ms_result.predict(start=inflation.index[0], end=inflation.index[-1])

plt.figure(figsize=(10, 4))
plt.plot(inflation.index, inflation, label='Observed Inflation')
plt.plot(inflation.index, predicted, label='Fitted Values', linestyle='--')
plt.title('In-sample Fit of the Markov-Switching Model')
plt.xlabel('Date')
plt.ylabel('Inflation (%)')
plt.legend()
plt.show()