import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


#rates = pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/10YBond.xlsx", parse_dates=[0], index_col=0)
rates=pd.read_excel(r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\10YBond.xlsx", parse_dates=[0], index_col=0)
rates.columns = ["interest_rate"]


"""
====================================================================================================
Estimating the parameters of the model
====================================================================================================
"""

# Define the time step for weekly data (assuming 52 weeks in a year)
dt = 1/52

# Create lagged series: r_t (current week) and r_t1 (next week)
r_t = rates[:-1]   # all observations except the last one
r_t1 = rates[1:]   # all observations except the first one

# Convert to numpy arrays for the regression
r_t = r_t.values
r_t1 = r_t1.values

# Prepare the independent variable for OLS by adding a constant (for the intercept)
X = sm.add_constant(r_t)

# Run OLS regression: r_{t+1} = theta + phi * r_t + error
model = sm.OLS(r_t1, X).fit()
print(model.summary())

# Extract estimated coefficients:
theta_hat = model.params[0]  # the intercept (θ)
phi_hat   = model.params[1]  # the coefficient on r_t (φ)

# Recover continuous-time parameters:
# Since φ = exp(-a * dt), then:
a_hat = -np.log(phi_hat) / dt  # annualized speed of mean reversion

# Given θ = b * (1 - φ), then:
b_hat = theta_hat / (1 - phi_hat)

# Estimate σ_η from the regression residuals:
sigma_eta_hat = np.std(model.resid, ddof=1)  # standard deviation of the residuals

# Recover the volatility σ:
sigma_hat = sigma_eta_hat * np.sqrt(2 * a_hat / (1 - phi_hat**2))

# Display the estimated parameters:
print(f"Estimated a (annualized mean reversion speed): {a_hat}")
print(f"Estimated b (long-term mean level): {b_hat}")
print(f"Estimated sigma (annualized volatility): {sigma_hat}")




"""
====================================================================================================
Simulating one path
====================================================================================================
"""

# Choose an initial interest rate (e.g., the most recent rate from your data)
r0 = 0.0235

# Set the simulation horizon (e.g., 10 years of weekly data)
num_years = 45
num_steps = int(num_years * 52)  # total number of weekly steps

# Pre-calculate constants for the simulation
exp_a_dt = np.exp(-a_hat * dt)
sigma_dt = sigma_hat * np.sqrt((1 - np.exp(-2 * a_hat * dt)) / (2 * a_hat))

# Initialize an array to store the simulated rates
simulated_rates = np.zeros(num_steps + 1)
simulated_rates[0] = r0

# Run the simulation: iterate over each time step
for t in range(1, num_steps + 1):
    # Draw a random sample from the standard normal distribution
    epsilon = np.random.normal()
    # Update the interest rate using the exact discretization formula
    simulated_rates[t] = (
        b_hat +
        (simulated_rates[t-1] - b_hat) * exp_a_dt +
        sigma_dt * epsilon
    )

# Plot the simulated interest rate path
plt.figure(figsize=(10, 6))
plt.plot(simulated_rates, label='Simulated Interest Rate')
plt.xlabel('Time (weeks)')
plt.ylabel('Interest Rate')
plt.title('Simulated Interest Rate Path using the Vasicek Model')
plt.legend()
plt.grid(True)
plt.show()



"""
====================================================================================================
Monte Carlo Simulation of the Vasicek Model
====================================================================================================
"""

r0 = 0.0235

# Simulation horizon (e.g., 10 years of weekly data)
num_years = 45
num_steps = int(num_years * 52)  # Total number of weekly steps

# Number of Monte Carlo simulation paths
num_paths = 10000

# -------------------------------
# Step 2: Pre-calculate constants for the exact discretization.
# The exact discretization of the Vasicek model is:
#   r_{t+1} = b + (r_t - b)*exp(-a*dt) + sigma*sqrt((1-exp(-2a*dt))/(2a))*epsilon_t,
# where epsilon_t ~ N(0,1)
exp_a_dt = np.exp(-a_hat * dt)
sigma_dt = sigma_hat * np.sqrt((1 - np.exp(-2 * a_hat * dt)) / (2 * a_hat))

# -------------------------------
# Step 3: Initialize an array to store simulated paths.
# The array dimensions are (num_steps+1) x num_paths.
simulated_paths = np.zeros((num_steps + 1, num_paths))
simulated_paths[0, :] = r0  # Set the initial rate for all paths

# -------------------------------
# Step 4: Run the simulation.
# We generate all the random draws at once for efficiency.
epsilons = np.random.normal(size=(num_steps, num_paths))

for t in range(1, num_steps + 1):
    simulated_paths[t, :] = (
        b_hat +
        (simulated_paths[t - 1, :] - b_hat) * exp_a_dt +
        sigma_dt * epsilons[t - 1, :]
    )

# -------------------------------
# Step 5: Visualization and Analysis

# Plot a few sample paths (e.g., 10 random paths)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(simulated_paths[:, i], lw=1.5, alpha=0.8, label=f'Path {i+1}')
plt.xlabel('Time (weeks)')
plt.ylabel('Interest Rate')
plt.title('Monte Carlo Simulation of Interest Rate Paths using the Vasicek Model')
plt.legend()
plt.grid(True)
plt.show()

# Optionally, you can calculate statistics across the paths:
# For example, compute the mean and percentiles at each time step:
mean_path = simulated_paths.mean(axis=1)
percentile_5 = np.percentile(simulated_paths, 5, axis=1)
percentile_95 = np.percentile(simulated_paths, 95, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(mean_path, label='Mean Path', color='black', lw=2)
plt.fill_between(range(num_steps + 1), percentile_5, percentile_95, color='gray', alpha=0.4,
                 label='5th-95th Percentile Range')
plt.xlabel('Time (weeks)')
plt.ylabel('Interest Rate')
plt.title('Monte Carlo Simulation: Mean and 5th-95th Percentile Range')
plt.legend()
plt.grid(True)
plt.show()
