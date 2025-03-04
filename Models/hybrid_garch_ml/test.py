import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load ETF data (with Danish column names) and set the date as index.
data = pd.read_csv(
    r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\iShares_Core_S&P_500_ETF_USD_Acc_EUR.csv",
    parse_dates=['Dato'], index_col='Dato')
data['log_returns'] = np.log(data["Slutkurs"] / data["Slutkurs"].shift(1))
returns = data["log_returns"].dropna()


def heston_log_likelihood(params, returns, dt=1 / 252):
    # Unpack parameters: mu, kappa, theta, sigma, rho, v0
    mu, kappa, theta, sigma, rho, v0 = params

    # Initialise log-likelihood
    log_lik = 0.0
    v_prev = v0

    # Loop over returns
    for r in returns:
        # Here you would use your chosen filter (e.g. EKF or Particle Filter) to estimate v_t
        # For illustration, we assume v_prev is our variance estimate at time t-1.

        # Compute the conditional mean and variance for the return
        # Mean is mu*dt, variance is v_prev * dt (approximation)
        cond_mean = mu * dt
        cond_var = v_prev * dt

        # Calculate the likelihood of observing return r
        # Assuming a normal density:
        likelihood = (1 / np.sqrt(2 * np.pi * cond_var)) * \
                     np.exp(- (r - cond_mean) ** 2 / (2 * cond_var))
        # Update log likelihood (avoid log(0) issues in practice)
        log_lik += np.log(max(likelihood, 1e-10))

        # Update the variance using the Heston variance equation (Euler discretisation)
        # A simple update for demonstration:
        v_new = v_prev + kappa * (theta - v_prev) * dt
        # Add the stochastic component – here, we ignore it or incorporate a filtered estimate
        v_prev = max(v_new, 1e-8)  # ensure positivity

    return -log_lik  # Negative log-likelihood for minimisation


# Initial guess for parameters: [mu, kappa, theta, sigma, rho, v0]
initial_params = [0.05, 1.5, 0.04, 0.3, -0.7, 0.04]

# Optimise using a minimisation routine
result = minimize(heston_log_likelihood, initial_params, args=(returns,), method='Nelder-Mead')

# Display calibrated parameters
print("Calibrated parameters:")
print(result.x)

results=result.x
mu=results[0]
kappa=results[1]
theta=results[2]
sigma=results[3]
rho=results[4]
v0=results[5]

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
T = 10  # total time in years
dt = 1 / 252  # daily time steps (252 trading days per year)
N = int(T / dt)  # total number of time steps (approx. 2520)
n_paths = 1000  # number of simulated paths

# Heston model parameters (example values – calibrate these for your data)
S0 = data["Slutkurs"].iloc[-1]  # initial price

# Pre-allocate arrays for simulation
S = np.zeros((N, n_paths))
v = np.zeros((N, n_paths))
S[0] = S0
v[0] = v0

# Time stepping via Euler discretisation
for t in range(1, N):
    # Generate two independent random normal variables
    Z1 = np.random.randn(n_paths)
    Z2 = np.random.randn(n_paths)
    # Create correlated random variables using Cholesky decomposition
    dW_S = np.sqrt(dt) * Z1
    dW_v = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

    # Ensure variance remains non-negative: Euler may require adjustments if v becomes negative
    v[t - 1] = np.maximum(v[t - 1], 0)

    # Update variance process (Heston SDE for v_t)
    v[t] = v[t - 1] + kappa * (theta - v[t - 1]) * dt + sigma * np.sqrt(v[t - 1]) * dW_v
    # Enforce non-negativity of variance
    v[t] = np.maximum(v[t], 0)

    # Update asset price process (Heston SDE for S_t)
    S[t] = S[t - 1] * (1 + mu * dt + np.sqrt(v[t - 1]) * dW_S)

# Plot a few sample paths for the ETF price over 10 years
plt.figure(figsize=(10, 6))
for i in range(100):  # plot 10 sample paths
    plt.plot(S[:, i], lw=0.8, alpha=0.8)
plt.title('Heston Model Simulation of ETF Price over 10 Years')
plt.xlabel('Time steps (Days)')
plt.ylabel('ETF Price')
plt.show()


mean_path=np.mean(S,axis=1)

plt.figure(figsize=(10, 6))
plt.plot(mean_path)
plt.show()

plt.hist(S[-1], bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
plt.show()