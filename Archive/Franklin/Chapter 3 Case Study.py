import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Define Functions =====================================================
# Forecast portfolio volatility
def sd_f(weight_vector, covar_table):
    covar_vector = np.dot(covar_table, weight_vector)
    return np.sqrt(np.dot(weight_vector, covar_vector))

# Expected portfolio return
def mean_f(weight_vector, return_vector):
    return np.dot(weight_vector, return_vector)

# Probability of goal achievement
def phi_f(goal_vector, goal_allocation, pool, mean, sd):
    required_return = (goal_vector[1] / (pool * goal_allocation))**(1 / goal_vector[2]) - 1
    if goal_allocation * pool >= goal_vector[1]:
        return 1
    else:
        return 1 - norm.cdf(required_return, loc=mean, scale=sd)

# Optimization objective function
def optim_function(weights, goal_vector, allocation, pool, return_vector, covar_table):
    return 1 - phi_f(goal_vector, allocation, pool,
                     mean_f(weights, return_vector),
                     sd_f(weights, covar_table))

# Constraint for weights to sum to 1
def constraint_function(weights):
    return np.sum(weights) - 1

# Mean-variance utility function
def mvu_f(weights, return_vector, covar_table, gamma):
    return -(mean_f(weights, return_vector) - 0.5 * gamma * sd_f(weights, covar_table)**2)

# Required return function
def r_req_f(goal_vector, goal_allocation, pool):
    return (goal_vector[1] / (goal_allocation * pool))**(1 / goal_vector[2]) - 1

# Load & Parse Data ====================================================
n_trials = 10**5

# Replace with actual file paths
goal_data_raw = pd.read_csv(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Franklin/Goals-Based Utility Practitioners Guide/Example Goal Details.csv")
capital_market_expectations_raw = pd.read_csv(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Franklin/Goals-Based Utility Practitioners Guide/Capital Market Expectations.csv")
correlations_raw = pd.read_csv(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Franklin/Goals-Based Utility Practitioners Guide/Correlations - Kitchen Sink.csv")

# Process data
num_assets = len(capital_market_expectations_raw)
num_goals = goal_data_raw.shape[1] - 1

return_vector = capital_market_expectations_raw.iloc[:, 1].values
correlations = correlations_raw.iloc[:num_assets, 1:num_assets + 1].values

# Build covariance matrix
vols = capital_market_expectations_raw.iloc[:, 2].values
covariances = np.outer(vols, vols) * correlations

# Parse goals
goal_A = goal_data_raw.iloc[:, 1].values
goal_B = goal_data_raw.iloc[:, 2].values
goal_C = goal_data_raw.iloc[:, 3].values
goal_D = goal_data_raw.iloc[:, 4].values
pool = 4654000

# STEP 1: Optimal Within-Goal Allocation =================================
goal_allocation = np.arange(0.01, 1.01, 0.01)
starting_weights = np.random.rand(num_assets)
starting_weights /= starting_weights.sum()

optimal_weights_A = []
optimal_weights_B = []
optimal_weights_C = []
optimal_weights_D = []

def optimise_goal(goal_vector):
    for alloc in goal_allocation:
        if goal_vector[1] <= pool * alloc:
            yield np.array([0] * (num_assets - 1) + [1])
        else:
            result = minimize(
                optim_function, starting_weights,
                args=(goal_vector, alloc, pool, return_vector, covariances),
                constraints={'type': 'eq', 'fun': constraint_function},
                bounds=[(0, 1) for _ in range(num_assets)],
            )
            yield result.x

optimal_weights_A = np.array(list(optimise_goal(goal_A)))
optimal_weights_B = np.array(list(optimise_goal(goal_B)))
optimal_weights_C = np.array(list(optimise_goal(goal_C)))
optimal_weights_D = np.array(list(optimise_goal(goal_D)))

# Log probabilities of achievement
phi_A = np.array([phi_f(goal_A, alloc, pool, mean_f(w, return_vector), sd_f(w, covariances)) for alloc, w in zip(goal_allocation, optimal_weights_A)])
phi_B = np.array([phi_f(goal_B, alloc, pool, mean_f(w, return_vector), sd_f(w, covariances)) for alloc, w in zip(goal_allocation, optimal_weights_B)])
phi_C = np.array([phi_f(goal_C, alloc, pool, mean_f(w, return_vector), sd_f(w, covariances)) for alloc, w in zip(goal_allocation, optimal_weights_C)])
phi_D = np.array([phi_f(goal_D, alloc, pool, mean_f(w, return_vector), sd_f(w, covariances)) for alloc, w in zip(goal_allocation, optimal_weights_D)])

# STEP 2: Optimal Across-Goal Allocation =================================
sim_goal_weights = np.random.dirichlet(np.ones(num_goals), n_trials) * 100
utility = (
    goal_A[0] * phi_A[np.round(sim_goal_weights[:, 0]).astype(int) - 1] +
    goal_B[0] * phi_B[np.round(sim_goal_weights[:, 1]).astype(int) - 1] +
    goal_C[0] * phi_C[np.round(sim_goal_weights[:, 2]).astype(int) - 1] +
    goal_D[0] * phi_D[np.round(sim_goal_weights[:, 3]).astype(int) - 1]
)

optimal_goal_weights = sim_goal_weights[np.argmax(utility)]

# Visualise Results =====================================================
plt.figure(figsize=(10, 6))
plt.plot(goal_allocation * 100, phi_A, label="Goal A", color="blue")
plt.plot(goal_allocation * 100, phi_B, label="Goal B", color="green")
plt.plot(goal_allocation * 100, phi_C, label="Goal C", color="orange")
plt.plot(goal_allocation * 100, phi_D, label="Goal D", color="red")
plt.xlabel("Goal Allocation (%)")
plt.ylabel("Probability of Achievement")
plt.title("Probability of Achievement vs Goal Allocation")
plt.legend()
plt.grid()
plt.show()
