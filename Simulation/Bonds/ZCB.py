import os
from codelib.file_management.dynamic_file_pathing import get_root
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from scipy import stats
import cvxpy as cp
from codelib.Johan.statistics import moments as mom
from codelib.Johan.portfolio_optimization import risk_metrics as rm
from codelib.Johan.portfolio_optimization import risk_metrics as rb
from codelib.Johan.portfolio_optimization import mean_variance as mvo
from codelib.Johan.fixed_income.cash_flows import CashFlow
from codelib.Johan.visualization.base import fan_chart

root = get_root()
data_folder = os.path.join(root, "GBI Optimisation", "Data")
output_folder = os.path.join(root, "Simulation", "Bonds")

np.random.seed(222)


"""
Initialize parameters for simulation
"""
# equity parameters
sigma = 0.005705482064922983
mu = 0.0001876482646574682

from codelib.Models.vasicek_model import VasicekModel
#short_rate = "/Users/osito/Repositories/AEF_thesis_GBI/Miscellaneous/Archive/Data/short_rate.xlsx"
short_rate = os.path.join(root, "Miscellaneous", "Archive", "Data", "short_rate.xlsx")

vas = VasicekModel(short_rate,252)
initial_rate=vas.rates.iloc[-1]
kappa, theta, beta = vas.estimate_params()

rp = -0.2

# correlation
rho = 0.6

# simulation definition
num_sim = 10_000
dt = 1.0 / 252
horizon = 10
num_time_steps = int(horizon / dt)
time_points = np.arange(0, num_time_steps + 1, 1) * dt


"""
Define functions for simulation
"""
def simulate_vasicek(initial_short_rate: float, kappa: float, theta: float, beta: float, horizon: float,
                     dt: float = 1.0 / 12, num_sim: int = 10000, z_mat=None):
    """
    simulates short rate processes in a vasicek setting until a given horizon

    Parameters
    ----------

    initial_short_rate:
        initial short rate
    kappa:
        speed of mean reversion.
    theta:
        long term mean of the short rate.
    dt:
        increments in time
    horizon:
        time until maturity/expiry (horizon).
    num_sim:
        number of simulations.
    """
    std_rates = np.sqrt(beta ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))

    num_periods = int(horizon / dt)
    short_rates = np.empty((num_sim, num_periods + 1))
    short_rates[:, 0] = initial_short_rate

    if z_mat is None:
        error_terms = np.random.normal(scale=std_rates, size=(num_sim, num_periods))
    else:
        error_terms = std_rates * z_mat

    for i in range(1, num_periods + 1):
        short_rates[:, i] = theta + (short_rates[:, i - 1] - theta) * np.exp(-kappa * dt) + error_terms[:, i - 1]

    return short_rates


def simulate_risk_drivers(mu: float, sigma: float,
                          initial_rate: float, kappa: float, theta: float, beta: float,
                          rho: float,
                          horizon: float,
                          dt: float = 1.0 / 12,
                          num_sim: int = 10_000):
    """
    Function simulating the risky asset and the short rate.
    """

    # define the number of time steps
    num_time_steps = int(horizon / dt)

    # convert parameters of equity values
    mu_scaled = (mu - 0.5 * sigma ** 2) * dt
    sigma_scaled = sigma * np.sqrt(dt)

    # define innovation correlation matrix
    z_corr_mat = np.array([[1.0, rho], [rho, 1.0]])

    # simulate innovations
    z_mat = np.random.multivariate_normal(np.zeros(2), z_corr_mat, size=(num_sim, num_time_steps))

    # simulate equity prices
    log_ret = mu_scaled + sigma_scaled * z_mat[:, :, 0]

    equity_prices = np.ones((num_sim, num_time_steps + 1))
    equity_prices[:, 1:] = np.exp(np.cumsum(log_ret, axis=1))

    # simulate short rates
    short_rates = simulate_vasicek(initial_short_rate=initial_rate,
                                   kappa=kappa,
                                   theta=theta,
                                   beta=beta,
                                   horizon=horizon,
                                   dt=dt,
                                   num_sim=num_sim,
                                   z_mat=z_mat[:, :, 1])

    return equity_prices, short_rates


equity_prices, short_rates = simulate_risk_drivers(mu, sigma,
                                                   initial_rate, kappa, theta, beta,
                                                   rho,
                                                   horizon,
                                                   dt,
                                                   num_sim)


"""
Plot fan chart of short rates
"""
percentiles = np.percentile(short_rates, [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
fan_chart(time_points, percentiles, ax=ax, color="navy")
plt.show()

short_rates = pd.DataFrame(short_rates)
short_rates.to_excel(os.path.join(output_folder, "short_rates_sim.xlsx"))






























