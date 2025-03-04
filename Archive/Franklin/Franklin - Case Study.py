"""import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ============================
# CLIENT GOALS
# ============================
goals = {
    "Children's Estate": {"amount": 5000000, "years": 30},
    "Living Expenses": {"amount": 5157000, "years": 10},
    "Vacation Home": {"amount": 713500, "years": 4},
    "Naming Rights": {"amount": 8812000, "years": 18},
}
wealth_available = 4654400

# Goal priority ranking
value_ratios = {
    "Living Expenses": 1.00,
    "Children's Estate": 0.45,
    "Vacation Home": 0.225,
    "Naming Rights": 0.1305,
}

# ============================
# CAPITAL MARKET EXPECTATIONS
# ============================
assets = ["Large Cap", "Mid Cap", "Small Cap", "Int'l Developed", "Emerging Markets",
          "US Agg Bond", "US High Yield", "US Treasury", "Corporate",
          "Gold", "Oil", "Private Equity", "Venture Capital", "Angel Venture", "Cash"]

expected_returns = np.array([0.09, 0.11, 0.12, 0.07, 0.09,
                             0.04, 0.06, 0.03, 0.05,
                             0.06, 0.04, 0.15, 0.16, -0.01, 0.01])

volatility = np.array([0.15, 0.16, 0.17, 0.15, 0.17,
                       0.05, 0.09, 0.03, 0.07,
                       0.19, 0.32, 0.28, 0.30, 0.82, 0.001])

k = len(assets)  # Number of asset classes
N = len(goals)   # Number of goals
varrho = 20      # Resolution level
tau = 10000       # Monte Carlo simulations

# ============================
# STAGE 1: WITHIN-GOAL OPTIMIZATION
# ============================
def lower_tail_cdf(Wj, G, t, omega, pi_func):
    """Computes the lower-tail CDF function (probability of not achieving goal)."""
    result = norm.cdf((Wj / (G * np.dot(omega, expected_returns)))**(1/t) - 1, *pi_func(omega))
    return float(result)

def objective_function(omega, Wj, G, t, pi_func):
    """Minimize downside risk for each goal."""
    penalty = 100 * (1 - np.sum(omega))**2  # Ensure sum(omega) = 1
    return lower_tail_cdf(Wj, G, t, omega, pi_func) + penalty

def weight_constraint(omega):
    return np.sum(omega) - 1

bounds = [(0, 1) for _ in range(k)]
constraints = {"type": "eq", "fun": weight_constraint}

# Solve for optimal portfolios
Omega_matrices = np.zeros((N, varrho, k))
goal_names = list(goals.keys())

for j, (goal, data) in enumerate(goals.items()):
    Wj, Tj = data["amount"], data["years"]
    for i in range(1, varrho + 1):
        theta_i = i / varrho
        omega_init = np.ones(k) / k
        result = minimize(objective_function, omega_init, args=(Wj, 1, Tj, lambda x: (0, 1)),
                          bounds=bounds, constraints=constraints, method="SLSQP")
        Omega_matrices[j, i - 1, :] = result.x

# ============================
# STAGE 2: ACROSS-GOAL ALLOCATION (Monte Carlo)
# ============================
def generate_theta(N):
    return np.random.dirichlet(np.ones(N))

def objective_function_across_goals(theta, W, G, Omega_matrices, value_ratios, pi_func):
    total_value = 0
    for j in range(N):
        row_index = int(theta[j] * (varrho - 1))
        investment_weights = Omega_matrices[j, row_index]
        failure_prob = lower_tail_cdf(W[j], G, goals[goal_names[j]]["years"], investment_weights, pi_func)
        total_value += value_ratios[goal_names[j]] * (1 - failure_prob)
    return total_value

# Monte Carlo optimization
best_theta = None
best_value = -np.inf

for i in range(tau):
    theta_candidate = generate_theta(N)
    value = objective_function_across_goals(theta_candidate, [goals[g]["amount"] for g in goal_names],
                                            1, Omega_matrices, value_ratios, lambda x: (0, 1))
    if value > best_value:
        best_value = value
        best_theta = theta_candidate

# Extract optimal investment weights
optimal_weights = np.zeros((N, k))
for j in range(N):
    row_index = int(best_theta[j] * (varrho - 1))
    optimal_weights[j, :] = Omega_matrices[j, row_index]

# ============================
# OUTPUT RESULTS
# ============================
df_theta = pd.DataFrame(best_theta, index=goal_names, columns=["Optimal Allocation"])
df_weights = pd.DataFrame(optimal_weights, index=goal_names, columns=assets)

df_theta.to_csv("optimal_theta.csv", index=True)
df_weights.to_csv("optimal_weights.csv", index=True)

print("Optimal Across-Goal Allocations:")
print(df_theta)

print("\nOptimal Investment Weights per Goal:")
print(df_weights)




import matplotlib.pyplot as plt

# Normalize goal allocations from 0 to 100 (mapping goals to a continuous scale)
goal_allocation = np.linspace(0, 100, len(df_weights))

# Create a stacked area chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(goal_allocation, df_weights.T, labels=df_weights.columns, alpha=0.8)

# Formatting the plot
plt.xlabel("Goal Allocation")
plt.ylabel("Investment Weight")
plt.title("Optimal Portfolios for Various Levels of Wealth Allocation")
plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show the plot
plt.show()"""









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ============================
# CLIENT GOALS
# ============================
goals = {
    "Children's Estate": {"amount": 5000000, "years": 30},
    "Living Expenses": {"amount": 5157000, "years": 10},
    "Vacation Home": {"amount": 713500, "years": 4},
    "Naming Rights": {"amount": 8812000, "years": 18},
}
wealth_available = 4654400

# Goal priority ranking (from elicited probabilities)
value_ratios = {
    "Living Expenses": 1.00,
    "Children's Estate": 0.45,
    "Vacation Home": 0.225,
    "Naming Rights": 0.1305,
}

# ============================
# CAPITAL MARKET EXPECTATIONS
# ============================
assets = ["Large Cap", "Mid Cap", "Small Cap", "Int'l Developed", "Emerging Markets",
          "US Agg Bond", "US High Yield", "US Treasury", "Corporate",
          "Gold", "Oil", "Private Equity", "Venture Capital", "Angel Venture", "Cash"]

expected_returns = np.array([0.09, 0.11, 0.12, 0.07, 0.09,
                             0.04, 0.06, 0.03, 0.05,
                             0.06, 0.04, 0.15, 0.16, -0.01, 0.01])

volatility = np.array([0.15, 0.16, 0.17, 0.15, 0.17,
                       0.05, 0.09, 0.03, 0.07,
                       0.19, 0.32, 0.28, 0.30, 0.82, 0.001])

k = len(assets)  # Number of asset classes
N = len(goals)  # Number of goals
varrho = 100  # Resolution level (higher = smoother allocation transitions)
tau = 1000  # Monte Carlo simulations


# ============================
# STAGE 1: WITHIN-GOAL OPTIMIZATION
# ============================
def lower_tail_cdf(Wj, G, t, omega):
    """Computes the lower-tail CDF function (probability of not achieving goal)."""
    result = norm.cdf((Wj / (G * np.dot(omega, expected_returns))) ** (1 / t) - 1, 0, 1)
    return float(result)


def objective_function(omega, Wj, G, t):
    """Minimize downside risk for each goal."""
    penalty = 100 * (1 - np.sum(omega)) ** 2  # Ensure sum(omega) = 1
    return lower_tail_cdf(Wj, G, t, omega) + penalty


def weight_constraint(omega):
    return np.sum(omega) - 1


bounds = [(0, 1) for _ in range(k)]
constraints = {"type": "eq", "fun": weight_constraint}

# Solve for optimal portfolios
Omega_matrices = np.zeros((N, varrho, k))
goal_names = list(goals.keys())

for j, (goal, data) in enumerate(goals.items()):
    Wj, Tj = data["amount"], data["years"]
    for i in range(varrho):
        theta_i = (i + 1) / varrho
        omega_init = np.ones(k) / k
        result = minimize(objective_function, omega_init, args=(Wj, 1, Tj),
                          bounds=bounds, constraints=constraints, method="SLSQP")
        Omega_matrices[j, i, :] = result.x


# ============================
# STAGE 2: ACROSS-GOAL ALLOCATION (Monte Carlo)
# ============================
def generate_theta(N):
    return np.random.dirichlet(np.ones(N))


def objective_function_across_goals(theta, W, G, Omega_matrices, value_ratios):
    total_value = 0
    for j in range(N):
        row_index = int(theta[j] * (varrho - 1))
        investment_weights = Omega_matrices[j, row_index]
        failure_prob = lower_tail_cdf(W[j], G, goals[goal_names[j]]["years"], investment_weights)
        total_value += value_ratios[goal_names[j]] * (1 - failure_prob)
    return total_value


best_theta = None
best_value = -np.inf

for i in range(tau):
    theta_candidate = generate_theta(N)
    value = objective_function_across_goals(theta_candidate, [goals[g]["amount"] for g in goal_names],
                                            1, Omega_matrices, value_ratios)
    if value > best_value:
        best_value = value
        best_theta = theta_candidate

# Extract optimal investment weights
optimal_weights = np.zeros((N, k))
for j in range(N):
    row_index = int(best_theta[j] * (varrho - 1))
    optimal_weights[j, :] = Omega_matrices[j, row_index]

# ============================
# PLOTTING STACKED AREA CHART
# ============================
goal_allocation = np.linspace(0, 100, varrho)

fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(goal_allocation, Omega_matrices[0].T, labels=assets, alpha=0.8)

plt.xlabel("Goal Allocation (%)")
plt.ylabel("Investment Weight")
plt.title("Optimal Portfolios for Various Levels of Wealth Allocation")
plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()


