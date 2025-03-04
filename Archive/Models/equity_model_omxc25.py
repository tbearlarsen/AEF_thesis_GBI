import numpy as np
import pandas as pd
from Simulation.Data.omxc25_data import close as prices
import matplotlib.pyplot as plt

"""
====================================================================================================
Estimating drift and volatility
====================================================================================================
"""

log_returns=np.log(prices/prices.shift(1)).dropna()

#Weekly parameters
mu_weekly=log_returns.mean()
sigma_weekly=log_returns.std()

#Annual parameters
mu_annual = mu_weekly * 52
sigma_annual = sigma_weekly * np.sqrt(52)

print(f"Estimated weekly drift: {mu_weekly:.4f}, Annual drift: {mu_annual:.4f}")
print(f"Estimated weekly volatility: {sigma_weekly:.4f}, Annual volatility: {sigma_annual:.4f}")


"""
====================================================================================================
Simulation of Asset Price Paths
====================================================================================================
"""

#Parameters
mu=mu_weekly
sigma=sigma_weekly
S0=prices.iloc[-1]

#Simulation setup
num_years=45
num_weeks=52*num_years
num_paths=10000
dt=1/52

simulated_prices=np.zeros((num_weeks+1,num_paths))
simulated_prices[0]=S0


#Monte Carlo Simulation
for t in range(1,num_weeks+1):
    epsilon=np.random.normal(0,1,num_paths)
    simulated_prices[t]=simulated_prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)


#Plot a few sample paths
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(simulated_prices[:, i], lw=1.5, alpha=0.8, label=f'Path {i+1}')
plt.xlabel('Time (weeks)')
plt.ylabel('Asset Price')
plt.title('Monte Carlo Simulation of Asset Price Paths using GBM')
plt.legend()
plt.grid(True)
plt.show()








