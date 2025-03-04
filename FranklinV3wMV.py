# %%
#!pip install numpy pandas scipy matplotlib

# %%
import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# %% [markdown]
# # Define Functions
# This section defines the functions used for calculating portfolio volatility, expected return, 
# goal achievement probability, and the objective (failure probability) to minimize.

# %%

def sd_f(weight_vector, covar_table):
    covar_vector = np.zeros(len(weight_vector))
    for z in range(len(weight_vector)):
        covar_vector[z] = np.sum(weight_vector * covar_table[:, z])
    return np.sqrt(np.sum(weight_vector * covar_vector))

# %%
def mean_f(weight_vector, return_vector):
    return np.sum(weight_vector * return_vector)

# %%
def phi_f(goal_vector, goal_allocation, pool, mean, sd):
    # goal_vector is [value ratio, funding requirement, time horizon]
    required_return = (goal_vector[1] / (pool * goal_allocation))**(1 / goal_vector[2]) - 1
    if goal_allocation * pool >= goal_vector[1]:
        return 1
    else:
        return 1 - norm.cdf(required_return, loc=mean, scale=sd)

# %%
def optim_function(weights):
    # Uses the current global variables: goal_vector, allocation, pool, return_vector, covar_table
    return 1 - phi_f(
        goal_vector,
        allocation,
        pool,
        mean_f(weights, return_vector),
        sd_f(weights, covar_table)
    )

# %%
def constraint_function(weights):
    # For SciPy equality constraints, we require constraint_function(weights) == 0.
    return np.sum(weights) - 1

# %%
def mvu_f(weights):
    # mvu_f is defined for mean-variance optimization (not used below).
    return -(mean_f(weights, return_vector) - 0.5 * gamma * sd_f(weights, covariances)**2)

# %%
def r_req_f(goal_vector, goal_allocation, pool):
    return (goal_vector[1] / (goal_allocation * pool))**(1 / goal_vector[2]) - 1

# %% [markdown]
# # Load & Parse Data

# %%

# Set number of Monte Carlo trials (matching R's 10^5)
n_trials = 10**5

# Load data from CSV files (adjust the paths as needed)
goal_data_raw = pd.read_csv(r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Example Goal Details.csv")
capital_market_expectations_raw = pd.read_csv(r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Franklin\Goals-Based Utility Practitioners Guide\Datasets\Capital Market Expectations_M.csv")
correlations_raw = pd.read_csv(r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Franklin\Goals-Based Utility Practitioners Guide\Datasets\Correlations_m.csv")

# Record number of potential investments and goals
num_assets = len(capital_market_expectations_raw.iloc[:, 1])
num_goals = goal_data_raw.shape[1] - 1

# Create vector of expected returns
return_vector = capital_market_expectations_raw.iloc[:, 1].to_numpy()

# Get the correlations as a numeric DataFrame
correlations = correlations_raw.iloc[:8, 1:9].astype(float)

# Build the covariance matrix
covariances = np.zeros((num_assets, num_assets))
for i in range(num_assets):
    for j in range(num_assets):
        covariances[i, j] = (
            capital_market_expectations_raw.iloc[i, 2] *
            capital_market_expectations_raw.iloc[j, 2] *
            correlations.iloc[i, j]
        )

# %% [markdown]
# # Parse Goal Data
# Each goal vector is of the form: [value ratio, funding requirement, time horizon]

# %%
goal_A = [
    goal_data_raw.iloc[0, 1],
    goal_data_raw.iloc[1, 1],
    goal_data_raw.iloc[2, 1]
]
goal_B = [
    goal_data_raw.iloc[0, 2],
    goal_data_raw.iloc[1, 2],
    goal_data_raw.iloc[2, 2]
]
goal_C = [
    goal_data_raw.iloc[0, 3],
    goal_data_raw.iloc[1, 3],
    goal_data_raw.iloc[2, 3]
]
goal_D = [
    goal_data_raw.iloc[0, 4],
    goal_data_raw.iloc[1, 4],
    goal_data_raw.iloc[2, 4]
]

# Total pool of wealth
pool = 4654000

# %% [markdown]
# # Step 1: Optimal Within-Goal Allocation
# Enumerate possible across-goal allocations (from 0.01 to 1)
# and, for each goal, optimize the subportfolio weights.

# %%

goal_allocation = np.arange(0.01, 1.01, 0.01)

# Starting weights (random initialization normalized to sum to 1)
starting_weights = np.random.uniform(0, 1, num_assets)
starting_weights /= np.sum(starting_weights)

# Initialize matrices to store the optimal weights for each goal
optimal_weights_A = np.zeros((len(goal_allocation), num_assets))
optimal_weights_B = np.zeros((len(goal_allocation), num_assets))
optimal_weights_C = np.zeros((len(goal_allocation), num_assets))
optimal_weights_D = np.zeros((len(goal_allocation), num_assets))

# %%
goal_allocation = np.arange(0.01, 1.01, 0.01)

# Starting weights (random initialization normalized to sum to 1)
starting_weights = np.random.uniform(0, 1, num_assets)
starting_weights /= np.sum(starting_weights)

# Initialize matrices to store the optimal weights for each goal
optimal_weights_A = np.zeros((len(goal_allocation), num_assets))
optimal_weights_B = np.zeros((len(goal_allocation), num_assets))
optimal_weights_C = np.zeros((len(goal_allocation), num_assets))
optimal_weights_D = np.zeros((len(goal_allocation), num_assets))

# Set SLSQP options to be more stringent, mimicking solnp's behavior.
slsqp_opts = {
    'ftol': 1e-12,     # function tolerance
    'eps': 1e-12,      # finite-difference step size for gradient estimation
    'maxiter': 10000,  # maximum iterations
    'disp': False     # do not display convergence messages
}

for i, alloc in enumerate(goal_allocation):
    allocation = alloc      # Global variable used in optim_function
    covar_table = covariances

    # Goal A Optimization
    goal_vector = goal_A   # Global variable used in optim_function
    if goal_A[1] <= pool * allocation:
        optimal_weights_A[i, :] = [0]*(num_assets - 1) + [1]
    else:
        result = minimize(
            optim_function,
            starting_weights,
            constraints=[{'type': 'eq', 'fun': constraint_function}],
            bounds=[(0, 1)] * num_assets,
            method='SLSQP',
            options=slsqp_opts
        )
        optimal_weights_A[i, :] = result.x

    # Goal B Optimization
    goal_vector = goal_B
    if goal_B[1] <= pool * allocation:
        optimal_weights_B[i, :] = [0]*(num_assets - 1) + [1]
    else:
        result = minimize(
            optim_function,
            starting_weights,
            constraints=[{'type': 'eq', 'fun': constraint_function}],
            bounds=[(0, 1)] * num_assets,
            method='SLSQP',
            options=slsqp_opts
        )
        optimal_weights_B[i, :] = result.x

    # Goal C Optimization
    goal_vector = goal_C
    if goal_C[1] <= pool * allocation:
        optimal_weights_C[i, :] = [0]*(num_assets - 1) + [1]
    else:
        result = minimize(
            optim_function,
            starting_weights,
            constraints=[{'type': 'eq', 'fun': constraint_function}],
            bounds=[(0, 1)] * num_assets,
            method='SLSQP',
            options=slsqp_opts
        )
        optimal_weights_C[i, :] = result.x

    # Goal D Optimization
    goal_vector = goal_D
    if goal_D[1] <= pool * allocation:
        optimal_weights_D[i, :] = [0]*(num_assets - 1) + [1]
    else:
        result = minimize(
            optim_function,
            starting_weights,
            constraints=[{'type': 'eq', 'fun': constraint_function}],
            bounds=[(0, 1)] * num_assets,
            method='SLSQP',
            options=slsqp_opts
        )
        optimal_weights_D[i, :] = result.x

# %%
# Calculate the best probability (phi) for each allocation level for every goal
phi_A = np.zeros(len(goal_allocation))
phi_B = np.zeros(len(goal_allocation))
phi_C = np.zeros(len(goal_allocation))
phi_D = np.zeros(len(goal_allocation))

for i, alloc in enumerate(goal_allocation):
    phi_A[i] = phi_f(goal_A, alloc, pool,
                     mean_f(optimal_weights_A[i, :], return_vector),
                     sd_f(optimal_weights_A[i, :], covariances))
    phi_B[i] = phi_f(goal_B, alloc, pool,
                     mean_f(optimal_weights_B[i, :], return_vector),
                     sd_f(optimal_weights_B[i, :], covariances))
    phi_C[i] = phi_f(goal_C, alloc, pool,
                     mean_f(optimal_weights_C[i, :], return_vector),
                     sd_f(optimal_weights_C[i, :], covariances))
    phi_D[i] = phi_f(goal_D, alloc, pool,
                     mean_f(optimal_weights_D[i, :], return_vector),
                     sd_f(optimal_weights_D[i, :], covariances))

# %% [markdown]
# # Step 2: Optimal Across-Goal Allocation
# Simulate goal weights and compute utility for each trial.

# %%
# Simulate goal weights: each row is a simulated allocation (in percentages)
sim_goal_weights = np.zeros((n_trials, num_goals), dtype=int)
for i in range(n_trials):
    rand_vector = np.random.uniform(0, 1, num_goals)
    normalizer = np.sum(rand_vector)
    # Compute rounded percentages and enforce a minimum of 1
    percents = np.round((rand_vector / normalizer) * 100, 0)
    sim_goal_weights[i, :] = np.maximum(percents, 1)


# Calculate utility for each simulated portfolio.
# Note: subtract 1 from simulated weights for 0-indexing.
utility = (
    goal_A[0] * phi_A[sim_goal_weights[:, 0] - 1] +
    goal_A[0] * goal_B[0] * phi_B[sim_goal_weights[:, 1] - 1] +
    goal_A[0] * goal_B[0] * goal_C[0] * phi_C[sim_goal_weights[:, 2] - 1] +
    goal_A[0] * goal_B[0] * goal_C[0] * goal_D[0] * phi_D[sim_goal_weights[:, 3] - 1]
)

# Find the index of the portfolio with the highest utility
index = np.argmax(utility)
optimal_goal_weights = sim_goal_weights[index, :]

# %% [markdown]
# # Step 3: Optimal Subportfolios & Aggregate Portfolio
# Retrieve the optimal subportfolio allocations and compute the aggregate portfolio.

# %%
# Retrieve optimal subportfolio allocations
optimal_subportfolios = np.zeros((num_goals, num_assets))
goals = ["A", "B", "C", "D"]

# For each goal, use the simulated percentage to select the corresponding row 
# from the optimal weights matrix (adjust for zero-indexing)
for i in range(num_goals):
    optimal_subportfolios[i, :] = eval(f"optimal_weights_{goals[i]}")[optimal_goal_weights[i] - 1, :]

# Compute the optimal aggregate investment portfolio.
optimal_aggregate_portfolio = (optimal_goal_weights / 100) @ optimal_subportfolios


# %% [markdown]
# # Visualize Results
# Plot the Goal A subportfolio allocation as a function of the across-goal allocation.

# %%

# Asset names from the first column of the capital market expectations
asset_names = capital_market_expectations_raw.iloc[:, 0].astype(str).tolist()

# Use stackplot to display the Goal A subportfolio allocation (stacked areas) versus goal allocation (in %)
plt.figure(figsize=(10, 6))
plt.stackplot(goal_allocation * 100, optimal_weights_A.T, labels=asset_names, alpha=0.7)
plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
plt.title("Goal A Subportfolio Allocation", fontsize=16, fontweight='bold')
plt.legend(title="Asset", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# %%
# Asset names from the first column of the capital market expectations
asset_names = capital_market_expectations_raw.iloc[:, 0].astype(str).tolist()

# Use stackplot to display the Goal A subportfolio allocation (stacked areas) versus goal allocation (in %)
plt.figure(figsize=(10, 6))
plt.stackplot(goal_allocation * 100, optimal_weights_B.T, labels=asset_names, alpha=0.7)
plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
plt.title("Goal B Subportfolio Allocation", fontsize=16, fontweight='bold')
plt.legend(title="Asset", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# %%
# Asset names from the first column of the capital market expectations
asset_names = capital_market_expectations_raw.iloc[:, 0].astype(str).tolist()

# Use stackplot to display the Goal A subportfolio allocation (stacked areas) versus goal allocation (in %)
plt.figure(figsize=(10, 6))
plt.stackplot(goal_allocation * 100, optimal_weights_C.T, labels=asset_names, alpha=0.7)
plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
plt.title("Goal C Subportfolio Allocation", fontsize=16, fontweight='bold')
plt.legend(title="Asset", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# %%
# Asset names from the first column of the capital market expectations
asset_names = capital_market_expectations_raw.iloc[:, 0].astype(str).tolist()

# Use stackplot to display the Goal A subportfolio allocation (stacked areas) versus goal allocation (in %)
plt.figure(figsize=(10, 6))
plt.stackplot(goal_allocation * 100, optimal_weights_D.T, labels=asset_names, alpha=0.7)
plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
plt.title("Goal D Subportfolio Allocation", fontsize=16, fontweight='bold')
plt.legend(title="Asset", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# # Print Results

# %%
print("Optimal Across-Goal Allocation:")
print(optimal_goal_weights)

print("\nOptimal Aggregate Investment Allocation:")
print(optimal_aggregate_portfolio)

# %%
# Create a DataFrame for the across-goal allocation.
df_across_goal = pd.DataFrame({
    "Goal": goals,
    "Allocation (%)": optimal_goal_weights
})

# Create a DataFrame for the aggregate portfolio.
# Multiply the weight by 100 to display percentages.
df_aggregate = pd.DataFrame({
    "Asset": asset_names,
    "Weight": optimal_aggregate_portfolio,
    "Allocation (%)": np.round(optimal_aggregate_portfolio * 100, 2)
})

print("Optimal Across-Goal Allocation:")
print(df_across_goal.to_string(index=False))

print("\nOptimal Aggregate Investment Allocation:")
print(df_aggregate.to_string(index=False))



# %% [markdown]
# %%
# # (Continuing from the end of your "no mean var" script...)
# # Step 4: Mean-Variance Frontier Construction

import matplotlib.pyplot as plt

# We'll vary gamma from 60 down to 1, just like the R code.
g_list = np.arange(60, 0, -1, dtype=float)

m_list = []  # Store resultant portfolio returns
s_list = []  # Store resultant portfolio standard deviations
optimal_weights_mv_list = []  # Store the optimal weights for each gamma

# Re-initialize random starting weights for building the frontier
starting_weights_mv = np.random.uniform(0, 1, num_assets)
starting_weights_mv /= np.sum(starting_weights_mv)

# Build the mean-variance frontier
for gamma_temp in g_list:

    # Define local objective so we can vary gamma
    def mvu_f_local(weights):
        return -(mean_f(weights, return_vector) - 0.5 * gamma_temp * sd_f(weights, covariances)**2)

    # Perform SLSQP to minimize the negative MV objective
    mv_result = minimize(
        mvu_f_local,
        starting_weights_mv,
        constraints=[{'type': 'eq', 'fun': constraint_function}],
        bounds=[(0, 1)] * num_assets,
        method='SLSQP',
        options=slsqp_opts
    )

    if mv_result.success:
        w_opt = mv_result.x
    else:
        # fallback: just store the last best known
        w_opt = starting_weights_mv

    optimal_weights_mv_list.append(w_opt)
    m_list.append(mean_f(w_opt, return_vector))
    s_list.append(sd_f(w_opt, covariances))

# Convert to arrays for plotting
m_list = np.array(m_list)
s_list = np.array(s_list)
optimal_weights_mv_array = np.array(optimal_weights_mv_list)

# Plot the mean-variance efficient frontier
plt.figure(figsize=(7, 5))
plt.plot(s_list, m_list, color="dodgerblue", linewidth=2)
plt.xlabel("Volatility", fontsize=14)
plt.ylabel("Return", fontsize=14)
plt.title("Mean-Variance Efficient Frontier", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# %%
# ### Identify 'last point' on the frontier (gamma = 0.01) and store it

gamma_small = 0.01

def mvu_f_small_gamma(weights):
    return -(mean_f(weights, return_vector) - 0.5 * gamma_small * sd_f(weights, covariances)**2)

last_res = minimize(
    mvu_f_small_gamma,
    starting_weights_mv,
    constraints=[{'type': 'eq', 'fun': constraint_function}],
    bounds=[(0, 1)] * num_assets,
    method='SLSQP',
    options=slsqp_opts
)

last_weights = last_res.x if last_res.success else starting_weights_mv
last_m = mean_f(last_weights, return_vector)

# %% [markdown]
# %%
# ### Step 5: Compute Probability of Success for Each Goal Using Mean-Variance "Fallback"

# In R, we built these arrays:
# mv_phi_A, mv_phi_B, mv_phi_C, mv_phi_D
# and we stored the 'fallback' subportfolio weights in
# optimal_mv_weights_A, ..._B, ..._C, ..._D.

mv_phi_A = np.zeros(len(goal_allocation))
mv_phi_B = np.zeros(len(goal_allocation))
mv_phi_C = np.zeros(len(goal_allocation))
mv_phi_D = np.zeros(len(goal_allocation))

# We'll also store the chosen portfolio weights for each goal-allocation pair
optimal_mv_weights_A = np.zeros((len(goal_allocation), num_assets))
optimal_mv_weights_B = np.zeros((len(goal_allocation), num_assets))
optimal_mv_weights_C = np.zeros((len(goal_allocation), num_assets))
optimal_mv_weights_D = np.zeros((len(goal_allocation), num_assets))

for i, alloc in enumerate(goal_allocation):

    # --- Goal A ---
    req_return_A = r_req_f(goal_A, alloc, pool)
    if req_return_A > last_m:
        # Use 'last_weights' from the frontier
        optimal_mv_weights_A[i, :] = last_weights
        mv_phi_A[i] = phi_f(
            goal_A, alloc, pool,
            mean_f(last_weights, return_vector),
            sd_f(last_weights, covariances)
        )
    else:
        # Use the goals-based weights from the no-mean-var solution
        optimal_mv_weights_A[i, :] = optimal_weights_A[i, :]
        mv_phi_A[i] = phi_f(
            goal_A, alloc, pool,
            mean_f(optimal_weights_A[i, :], return_vector),
            sd_f(optimal_weights_A[i, :], covariances)
        )

    # --- Goal B ---
    req_return_B = r_req_f(goal_B, alloc, pool)
    if req_return_B > last_m:
        optimal_mv_weights_B[i, :] = last_weights
        mv_phi_B[i] = phi_f(
            goal_B, alloc, pool,
            mean_f(last_weights, return_vector),
            sd_f(last_weights, covariances)
        )
    else:
        optimal_mv_weights_B[i, :] = optimal_weights_B[i, :]
        mv_phi_B[i] = phi_f(
            goal_B, alloc, pool,
            mean_f(optimal_weights_B[i, :], return_vector),
            sd_f(optimal_weights_B[i, :], covariances)
        )

    # --- Goal C ---
    req_return_C = r_req_f(goal_C, alloc, pool)
    if req_return_C > last_m:
        optimal_mv_weights_C[i, :] = last_weights
        mv_phi_C[i] = phi_f(
            goal_C, alloc, pool,
            mean_f(last_weights, return_vector),
            sd_f(last_weights, covariances)
        )
    else:
        optimal_mv_weights_C[i, :] = optimal_weights_C[i, :]
        mv_phi_C[i] = phi_f(
            goal_C, alloc, pool,
            mean_f(optimal_weights_C[i, :], return_vector),
            sd_f(optimal_weights_C[i, :], covariances)
        )

    # --- Goal D ---
    req_return_D = r_req_f(goal_D, alloc, pool)
    if req_return_D > last_m:
        optimal_mv_weights_D[i, :] = last_weights
        mv_phi_D[i] = phi_f(
            goal_D, alloc, pool,
            mean_f(last_weights, return_vector),
            sd_f(last_weights, covariances)
        )
    else:
        optimal_mv_weights_D[i, :] = optimal_weights_D[i, :]
        mv_phi_D[i] = phi_f(
            goal_D, alloc, pool,
            mean_f(optimal_weights_D[i, :], return_vector),
            sd_f(optimal_weights_D[i, :], covariances)
        )

# %% [markdown]
# %%
# ### Step 6: Across-Goal Allocation (Mean-Variance Version)
# We reuse the same `sim_goal_weights` from before but compute a new utility: `utility_mv`.

utility_mv = (
    goal_A[0] * mv_phi_A[sim_goal_weights[:, 0] - 1] +
    goal_A[0] * goal_B[0] * mv_phi_B[sim_goal_weights[:, 1] - 1] +
    goal_A[0] * goal_B[0] * goal_C[0] * mv_phi_C[sim_goal_weights[:, 2] - 1] +
    goal_A[0] * goal_B[0] * goal_C[0] * goal_D[0] * mv_phi_D[sim_goal_weights[:, 3] - 1]
)

index_mv = np.argmax(utility_mv)
optimal_goal_weights_mv = sim_goal_weights[index_mv, :]

print("Optimal Across-Goal Allocation (Mean-Variance Version):")
print(optimal_goal_weights_mv)

# %% [markdown]
# %%
# ### Step 7: Visualize the Results for the Mean-Variance Subportfolios
# For instance, let's plot Goal A's subportfolio weights under the mean-variance approach.

plt.figure(figsize=(10, 6))
plt.stackplot(
    goal_allocation * 100,
    optimal_mv_weights_A.T,
    labels=asset_names,
    alpha=0.7
)
plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
plt.title("Goal A Subportfolio Allocation (Mean-Variance Logic)", fontsize=16, fontweight='bold')
plt.legend(title="Asset", fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ### Step 8: Compare Probability of Achievement (No-Mean-Var vs. Mean-Var) for Goal A
# We'll make a simple line plot of `phi_A` and `mv_phi_A` across the range of allocations.
# %%

plt.figure(figsize=(8, 5))
plt.plot(goal_allocation * 100, phi_A, label="Goals-Based Only", linewidth=2)
plt.plot(goal_allocation * 100, mv_phi_A, label="Mean-Var Logic", linewidth=2)
plt.xlabel("Goal Allocation (%)", fontsize=14)
plt.ylabel("Probability of Achievement", fontsize=14)
plt.title("Goal A: Probability of Achievement Comparison", fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
