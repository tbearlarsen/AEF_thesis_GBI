#%% md
# V1 (02 11 2025): Translated from R to Python and added the visualizations for goals B to D
# 
# 
# V2 (04 01 2025): Loop system implemented
# 
# V2.5 (04 02 2025): Loops system tweaks. Removed openpyxl as it was corrupting files and replaced it with xlwings
# 
# V3 & V4 finalize the loop except for the tax and asset channel
# 
# V 5,6,7 add tax shenanigans. V7C added the fully functioning inflow and outflow of wealth with the returns and taxes in it.
# 
# V8 modifies the aktiesparekonto pool so that it is increased when taxes are payed.
# 
# V9 substracts from the wealth pool the amount for the goals reached. It does not just substract the full amount of the goal, but rather the % from the allocation of the year prior to the goal being handed.
# 
# V10 Corrects returns to be inflation adjusted
# 
# V11 Opportunity cost calculation to redeem goal
# 
# V12 FinalWealth works without excel
# 
# V13 Loop check works without excel
# 
# 
#%%
#!pip install numpy pandas scipy matplotlib xlwings
#%%
import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import subprocess

#%%
# -- Variable Setup -- ##

#Monte Carlo Trials
n_trials = 10**5

# -V10 - Static Inflation
inflation_rate = 0.02  # 2%
#Case Study Profile Selection
Profile = "P1" #Either P1 or P2

#Excel worksheets
excel_returns = "Returns"
excel_volatilities = "Volatilities"
excel_correlation = "Correlation"
excel_income = "Salary"

#%% md
# # Define Functions
# This section defines the functions used for calculating portfolio volatility, expected return, 
# goal achievement probability, and the objective (failure probability) to minimize.
#%%

def sd_f(weight_vector, covar_table):
    covar_vector = np.zeros(len(weight_vector))
    for z in range(len(weight_vector)):
        covar_vector[z] = np.sum(weight_vector * covar_table[:, z])
    return np.sqrt(np.sum(weight_vector * covar_vector))
#%%
def mean_f(weight_vector, return_vector):
    return np.sum(weight_vector * return_vector)
#%%
def phi_f(goal_vector, goal_allocation, pool, mean, sd):
    # goal_vector is [value ratio, funding requirement, time horizon]
    required_return = (goal_vector[1] / (pool * goal_allocation))**(1 / goal_vector[2]) - 1
    if goal_allocation * pool >= goal_vector[1]:
        return 1
    else:
        return 1 - norm.cdf(required_return, loc=mean, scale=sd)
#%%
def optim_function(weights):
    # Uses the current global variables: goal_vector, allocation, pool, return_vector, covar_table
    return 1 - phi_f(
        goal_vector,
        allocation,
        pool,
        mean_f(weights, return_vector),
        sd_f(weights, covar_table)
    )
#%%
def constraint_function(weights):
    # For SciPy equality constraints, we require constraint_function(weights) == 0.
        return np.sum(weights) - 1
#%%
def mvu_f(weights):
    # mvu_f is defined for mean-variance optimization (not used below).
    return -(mean_f(weights, return_vector) - 0.5 * gamma * sd_f(weights, covariances)**2)
#%%
def r_req_f(goal_vector, goal_allocation, pool):
    return (goal_vector[1] / (goal_allocation * pool))**(1 / goal_vector[2]) - 1
#%%
# V 11 Utility Loss function for cashing out goal
def net_utility_change_from_goal_funding(
    goal_value_ratio,
    goal_required,
    portfolio_value,
    future_goals,
    future_goal_weights,
    return_vector,
    covariances
):
    # Portfolio value *after* cashing out the current goal
    portfolio_after = portfolio_value - goal_required

    # Utility loss from hitting future goals with reduced capital
    future_utility_loss = 0
    phi_details = []

    for i, goal in enumerate(future_goals):
        alloc = future_goal_weights[i]
        mean = mean_f(alloc, return_vector)
        sd = sd_f(alloc, covariances)

        phi_with = phi_f(goal, 1, portfolio_after, mean, sd)
        phi_without = phi_f(goal, 1, portfolio_value, mean, sd)
        value_weight = goal[0]

        future_utility_loss += value_weight * (phi_without - phi_with)
        phi_details.append((phi_without, phi_with))

    # Utility gain from achieving current goal
    utility_gain_now = value_now_nested * 1  # goal fully achieved if funded

    # Net effect on utility
    net_utility_change = utility_gain_now - future_utility_loss

    return net_utility_change, future_utility_loss, phi_details

#%%
# V11

def nested_value_ratio(goal_index, goal_value_list):
    ratio = 1.0
    for i in range(goal_index + 1):
        ratio *= goal_value_list[i]
    return ratio

#%%
def get_goal_data(master_excel_path, plan=Profile):
    """
    Returns a DataFrame of the specified goal table (P1 or P2).
    P1 => B2:F5
    P2 => B7:F10
    """
    if plan == "P1":
        skip = 1  # start reading at row 2
    elif plan == "P2":
        skip = 6  # start reading at row 7
    else:
        raise ValueError("Plan not recognized. Use 'P1' or 'P2'.")

    df_goals = pd.read_excel(master_excel_path,sheet_name="Goals",skiprows=skip,nrows=4,usecols="B:F",header=0)
    # First column is "Goal Info", so make that the index
    df_goals.set_index(df_goals.columns[0], inplace=True)
    return df_goals


#%% md
# # Load & Parse Data
#%%

## -- Repo Root and Folders -- ##

# Get repo root and set folders
root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
data_folder = os.path.join(root, "GBI Optimisation", "data")
output_folder = os.path.join(root, "GBI Optimisation")

# Get excel file and sheets
master_excel_path = os.path.join(data_folder, "Master.xlsx")

df_returns = pd.read_excel(master_excel_path, sheet_name=excel_returns)
df_vols = pd.read_excel(master_excel_path, sheet_name=excel_volatilities)

df_returns.set_index(df_returns.columns[0], inplace=True)
df_vols.set_index(df_vols.columns[0], inplace=True)

df_corr = pd.read_excel(master_excel_path, sheet_name=excel_correlation)
df_corr.set_index(df_corr.columns[0], inplace=True)

#%%
# - NEW V8 CONTENT REGARDING AKTIESPAREKONTO ALLOCATION - #

# Initial cap in 2025
initial_ask_cap = 166200
growth_rate = 0.1265  # 12.65% annual increase
aktiesparekonto_used_total = 0

#V12 moved finalwealth to python
final_wealth_tracker = {}

#V15 goal info stored in python instead of excel
goal_weights_tracker = []          # Stores goal weights for each year
aggregate_portfolio_tracker = []   # Stores aggregate portfolio for each year

# V15: Store all optimal weight matrices by year
goal_optimal_weights_A = {}
goal_optimal_weights_B = {}
goal_optimal_weights_C = {}
goal_optimal_weights_D = {}

#%%
# V 13 - Year Loop table
# -- Replaces Excel loop sheet with in-code structure -- #
start_year = 2025
end_year = 2075
N_max = end_year - start_year + 1

age_p1_start = 25
age_p2_start = 40

loop_df = pd.DataFrame({
    "N": range(1, N_max + 1),
    "Year": list(range(start_year, end_year + 1)),
    "AgeP1": [age_p1_start + i for i in range(N_max)],
    "AgeP2": [age_p2_start + i for i in range(N_max)],
    "LoopStatus": ['N'] * N_max
})

#%%
# V14 - Load salary data for all years
df_salary = pd.read_excel(master_excel_path, sheet_name=excel_income)
df_salary.set_index(df_salary.columns[0], inplace=True)
#%%
asset_level_log = []  # NEW: Log each asset's return pre- and post-tax by account

while True:
    pending_rows = loop_df[loop_df["LoopStatus"] == "N"]

    # If we find any rows, get the row with the lowest Year
    if pending_rows.empty:
        print("All loops completed")
        break


    chosen_row = pending_rows.loc[pending_rows["Year"].idxmin()]
    loop_year = chosen_row["Year"]
    loop_number = chosen_row["N"]
    loop_age1 = chosen_row["AgeP1"]
    loop_age2 = chosen_row["AgeP2"]

    print(loop_year, loop_number, loop_age1, loop_age2)

    # - Wealth - #

    # Get salary for current year
    salary = df_salary.loc[Profile, str(loop_year)]


    # - Dynamic time horizon based on current year - #

    # Step 1: Get Starting Year (minimum year in Loop sheet)
    starting_year = start_year
    prev_year = str(loop_year - 1)

    # If first loop year, no previous wealth exists
    if loop_year == starting_year:
        prev_wealth = 0
        pool = salary
    else:
        try:
            prev_wealth = final_wealth_tracker[int(prev_year)]
        except KeyError:
            raise KeyError(f"Previous wealth not found for {Profile} in {prev_year}")
        pool = salary + prev_wealth

    capital_market_expectations_raw = {}
    for asset in df_returns.index:
        expected_return = df_returns.loc[asset, str(loop_year)]
        volatility = df_vols.loc[asset, 'volatility']
        capital_market_expectations_raw[asset] = {
            'Return Forecast': expected_return,
            'Volatility Forecast': volatility
        }

    capital_market_expectations_raw = pd.DataFrame.from_dict(capital_market_expectations_raw, orient='index')

    capital_market_expectations_raw = capital_market_expectations_raw.reset_index()
    capital_market_expectations_raw.rename(columns={'index': 'Unnamed: 0'}, inplace=True)

    # Rearrange columns to match your old format (optional):
    capital_market_expectations_raw = capital_market_expectations_raw[['Unnamed: 0', 'Return Forecast', 'Volatility Forecast']]

    goal_data_raw = get_goal_data(master_excel_path, plan="P1")

    # - Dynamic time horizon based on current year - #


    # Step 2: Compute Goal Years = starting_year + time_horizon
    goal_horizons = goal_data_raw.loc["Time Horizon"].astype(int)
    goal_years = starting_year + goal_horizons

    # Step 3: Recalculate Time Horizons = goal_years - current loop_year
    adjusted_horizons = goal_years - loop_year

    # Step 4: Replace the "Time Horizon" row in goal_data_raw
    goal_data_raw.loc["Time Horizon"] = adjusted_horizons

    # - V8 - Dynamically compute ASK cap per year
    aktiesparekonto_cap = initial_ask_cap * ((1 + growth_rate) ** (loop_year - starting_year))

    goals = ["A", "B", "C", "D"]
    active_goal_mask = np.array([adjusted_horizons[f"GOAL {g}"] > 0 for g in goals])

    # Optional: print to confirm
    print("Starting Year:", starting_year)
    print("Current Year:", loop_year)
    print("Goal Years:", goal_years.to_dict())
    print("Adjusted Horizons:", adjusted_horizons.to_dict())



    # Record number of potential investments and goals
    num_assets = capital_market_expectations_raw.shape[0]
    num_goals = goal_data_raw.shape[1]

    # Create vector of expected returns
    return_vector = (capital_market_expectations_raw["Return Forecast"] - inflation_rate).to_numpy()

    # Get the correlations as a numeric DataFrame (just a num_assets × num_assets block)
    correlations = df_corr.iloc[:num_assets, :num_assets].astype(float)

    # Build the covariance matrix: stdev_i * stdev_j * correlation_ij
    stdevs = capital_market_expectations_raw["Volatility Forecast"].to_numpy()
    covariances = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(num_assets):
            covariances[i, j] = stdevs[i] * stdevs[j] * correlations.iloc[i, j]

    goal_A = goal_data_raw["GOAL A"].values
    goal_B = goal_data_raw["GOAL B"].values
    goal_C = goal_data_raw["GOAL C"].values
    goal_D = goal_data_raw["GOAL D"].values

    # - Optimal Goal Allocation - #


    goal_allocation = np.arange(0.01, 1.01, 0.01)

    # Starting weights (random initialization normalized to sum to 1)
    starting_weights = np.random.uniform(0, 1, num_assets)
    starting_weights /= np.sum(starting_weights)

    # Initialize matrices to store the optimal weights for each goal
    optimal_weights_A = np.zeros((len(goal_allocation), num_assets))
    optimal_weights_B = np.zeros((len(goal_allocation), num_assets))
    optimal_weights_C = np.zeros((len(goal_allocation), num_assets))
    optimal_weights_D = np.zeros((len(goal_allocation), num_assets))

    goal_allocation = np.arange(0.01, 1.01, 0.01)

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

    goal_optimal_weights_A[loop_year] = optimal_weights_A
    goal_optimal_weights_B[loop_year] = optimal_weights_B
    goal_optimal_weights_C[loop_year] = optimal_weights_C
    goal_optimal_weights_D[loop_year] = optimal_weights_D

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

    # Simulate goal weights: each row is a simulated allocation (in percentages)
    sim_goal_weights = np.random.multinomial(100, [1/num_goals]*num_goals, size=n_trials) #this one sums to 100 so its good
    for i in range(n_trials):
        rand_vector = np.random.uniform(0, 1, num_goals)
        normalizer = np.sum(rand_vector)
        percents = np.round((rand_vector / normalizer) * 100, 0)

        # Only enforce floor for active goals
        floor_applied = np.where(active_goal_mask, np.maximum(percents, 1), percents)
        sim_goal_weights[i, :] = floor_applied


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

    # - Optimal Subportfolio Allocation - #

    # Retrieve optimal subportfolio allocations
    optimal_subportfolios = np.zeros((num_goals, num_assets))

    # For each goal, use the simulated percentage to select the corresponding row
    # from the optimal weights matrix (adjust for zero-indexing)
    for i in range(num_goals):
        optimal_subportfolios[i, :] = eval(f"optimal_weights_{goals[i]}")[optimal_goal_weights[i] - 1, :]

    # Compute the optimal aggregate investment portfolio.
    optimal_aggregate_portfolio = (optimal_goal_weights / 100) @ optimal_subportfolios

    #Define asset names
    asset_names = capital_market_expectations_raw.iloc[:, 0].astype(str).tolist()

    # - Storing and exporting results - #

    # Create a DataFrame for the aggregate portfolio.
    # First calculate unrounded percentages
    raw_alloc = optimal_aggregate_portfolio * 100

    # Normalize to force sum = 100 after rounding
    normalized_alloc = raw_alloc / raw_alloc.sum() * 100
    normalized_goal_alloc = np.zeros_like(optimal_goal_weights, dtype=float)
    active_sum = np.sum(optimal_goal_weights[active_goal_mask])
    normalized_goal_alloc[active_goal_mask] = (optimal_goal_weights[active_goal_mask] / active_sum) * 100

    normalized_weights = normalized_alloc / 100  # convert back to decimal weights

    # Create a DataFrame for the across-goal allocation.
    df_across_goal = pd.DataFrame({
        "Goal": goals,
        "Allocation (%)": np.round(normalized_goal_alloc, 2) # Keep raw percentages
    })

    # Round after normalization
    df_aggregate = pd.DataFrame({
        "Asset": asset_names,
        "Weight": normalized_weights,
        "Invested Amt (DKK)": normalized_weights * pool,
        "Allocation (%)": np.round(normalized_alloc, 2)  # keep for display only
    })


    # - NEW V5 CONTENT REGARDING AKTIESPAREKONTO ALLOCATION - #
    # --- NEW: Exact and Proportional ASK Logic ---
    account_allocations = []
    equity_rows = []
    non_equity_rows = []

    # First pass: split equity and non-equity rows
    for i, row in df_aggregate.iterrows():
        asset_name = row["Asset"]
        invested = row["Invested Amt (DKK)"]
        weight = row["Weight"]

        row_dict = {
            "Year": loop_year,
            "Asset": asset_name,
            "Invested": invested,
            "Weight": weight,
            "ASK (DKK)": 0,
            "Normal (DKK)": 0,
        }

        if "Equities" in asset_name:
            equity_rows.append(row_dict)
        else:
            row_dict["Normal (DKK)"] = invested
            non_equity_rows.append(row_dict)

    # Determine total equity to allocate proportionally if ASK cap remains
    total_equity = sum(row["Invested"] for row in equity_rows)
    remaining_ask_cap = max(0, aktiesparekonto_cap - aktiesparekonto_used_total)
    ask_fraction = min(1, remaining_ask_cap / total_equity) if total_equity > 0 else 0

    # Apply proportional ASK allocation to equities
    for row in equity_rows:
        ask_part = row["Invested"] * ask_fraction
        normal_part = row["Invested"] - ask_part
        row["ASK (DKK)"] = round(ask_part, 2)
        row["Normal (DKK)"] = round(normal_part, 2)
        aktiesparekonto_used_total += ask_part

    # Combine and finalize
    account_allocations = equity_rows + non_equity_rows


    df_accounts = pd.DataFrame(account_allocations)
    print(f"[DEBUG] Final weight sum: {sum(row['Weight'] for row in account_allocations):.10f}")
    # - V5 END - #


    # - NEW V6 CONTENT REGARDING GAINS & TAXES - #


    share_income_normal = 0  # for progressive tax
    tax_ask = 0  # Track ASK tax separately

    # Update portfolio and compute gains
    for row in account_allocations:
        asset = row["Asset"]
        ask = row["ASK (DKK)"]
        normal = row["Normal (DKK)"]
        weight = row["Weight"]
        expected_return_nominal = capital_market_expectations_raw.loc[
            capital_market_expectations_raw["Unnamed: 0"] == asset,
            "Return Forecast"
        ].values[0]
        expected_return_inflation_adj = (
            capital_market_expectations_raw.loc[
                capital_market_expectations_raw["Unnamed: 0"] == asset,
                "Return Forecast"
            ].values[0] - inflation_rate
        )

        # V7C -LOG ASSET RETURNS BY ACCOUNT
        if ask > 0:
            new_val = ask * (1 + expected_return_inflation_adj)
            gain = new_val - ask
            tax = 0.17 * gain
            asset_level_log.append({
                "Year": loop_year,
                "Account": "ASK",
                "Asset": asset,
                "Invested": ask,
                "Weight": ask / pool,
                "Return (Nominal)": expected_return_nominal,
                "Inflation Rate": inflation_rate,
                "Return (Inflation Adjusted)": expected_return_inflation_adj,
                "Gross Gain": gain,
                "Tax": tax,
                "Net Gain": gain - tax,
                "End Value": ask + gain - tax,
                "TOTAL ASK Cap (DKK)": aktiesparekonto_cap
            })

        if normal > 0:
            new_val = normal * (1 + expected_return_inflation_adj)
            gain = new_val - normal
            share_income_normal += gain
            asset_level_log.append({
                "Year": loop_year,
                "Account": "NORMAL",
                "Asset": asset,
                "Invested": normal,
                "Weight": normal / pool,
                "Return (Nominal)": expected_return_nominal,
                "Inflation Rate": inflation_rate,
                "Return (Inflation Adjusted)": expected_return_inflation_adj,
                "Gross Gain": gain,
                "Tax": None,  # progressive tax logged later
                "Net Gain": None,
                "End Value": None,
                "TOTAL ASK Cap (DKK)": aktiesparekonto_cap
            })

            #V7C END

    # Tax on Normal Account (progressive)
    cap = 67500
    if share_income_normal <= cap:
        tax_normal = 0.27 * share_income_normal
    else:
        tax_normal = 0.27 * cap + 0.42 * (share_income_normal - cap)

    # Distribute tax proportionally across NORMAL assets
    normal_log_rows = [
        row for row in asset_level_log
        if row["Account"] == "NORMAL" and row["Year"] == loop_year
    ]
    total_normal_gain = sum(row["Gross Gain"] for row in normal_log_rows)

    print(f"[DEBUG] Year: {loop_year}")
    print(f"[DEBUG] Gross Gains (Normal account): {share_income_normal:.2f}")
    print(f"[DEBUG] Tax Calculated (Normal account): {tax_normal:.2f}")

    for row in normal_log_rows:
        if total_normal_gain > 0:
            share = row["Gross Gain"] / total_normal_gain
            tax = share * tax_normal
        else:
            tax = 0

        row["Tax"] = tax
        row["Net Gain"] = row["Gross Gain"] - tax
        row["End Value"] = row["Invested"] + row["Net Gain"]

    total_gains_normal = share_income_normal - tax_normal


    print("Optimal Across-Goal Allocation:")
    print(df_across_goal.to_string(index=False))

    print("\nOptimal Aggregate Investment Allocation:")
    print(df_aggregate.to_string(index=False))

    print("\nUtility:")
    print(utility)

    print("\nProbability of Achieving Each Goal at Optimal Allocation:")
    print(f"Goal A: {phi_A[optimal_goal_weights[0] - 1]:.4f}")
    print(f"Goal B: {phi_B[optimal_goal_weights[1] - 1]:.4f}")
    print(f"Goal C: {phi_C[optimal_goal_weights[2] - 1]:.4f}")
    print(f"Goal D: {phi_D[optimal_goal_weights[3] - 1]:.4f}")

    # -- V15 Goal weights -- #

    # Store current year's across-goal allocation
    goal_weights_tracker.append({
        "Year": loop_year,
        **{f"Goal {goals[i]} (%)": normalized_goal_alloc[i] for i in range(num_goals)}
    })

    # Store current year's aggregate portfolio
    for i in range(num_assets):
        aggregate_portfolio_tracker.append({
            "Year": loop_year,
            "Asset": asset_names[i],
            "Weight": normalized_weights[i],
            "Allocation (%)": np.round(normalized_alloc[i], 2),
            "Invested Amt (DKK)": normalized_weights[i] * pool
        })


    # -- V13 - Modify Loop Check -- #
    print("Starting loop status check (in-memory)...")

    # Show up to 10 rows (adjust as needed) of the current loop status
    preview_rows = loop_df.tail(10)
    for i, row in preview_rows.iterrows():
        print(f"Row {i+2}: Year = {row['Year']}, Status = {row['LoopStatus']}")

    # Update the current year's loop status
    loop_df.loc[loop_df["Year"] == loop_year, "LoopStatus"] = "Y"
    print(f"✅ LoopStatus updated to 'Y' for Year {loop_year}")

    this_year_rows = [row for row in asset_level_log if row["Year"] == loop_year]
    end_value_sum = sum(row["End Value"] for row in this_year_rows)

    # - V9 - If a goal has 1 year left, annotate allocation & amount taken in asset_level_log --- #
    goal_payout_info = {}

    for i, goal in enumerate(goals):
        if adjusted_horizons[f"GOAL {goal}"] == 1:
            alloc_pct = normalized_goal_alloc[i]
            goal_required = goal_data_raw.loc["Funding Requirement", f"GOAL {goal}"]
            amount_alloc_based = (alloc_pct / 100) * end_value_sum

            future_goals = [
                goal_data_raw[f"GOAL {g}"].values
                for j, g in enumerate(goals)
                if g != goal and adjusted_horizons[f"GOAL {g}"] > 1
            ]
            future_weights = [
                optimal_subportfolios[j, :]
                for j, g in enumerate(goals)
                if g != goal and adjusted_horizons[f"GOAL {g}"] > 1
            ]

            value_now_nested = nested_value_ratio(i, [goal_data_raw.loc["Value Ratio", f"GOAL {g}"] for g in goals])

            net_util, fut_loss, phi_details = net_utility_change_from_goal_funding(
                value_now_nested,
                goal_required,
                end_value_sum,
                future_goals,
                future_weights,
                return_vector,
                covariances
            )

            if net_util > 0:
                amount_taken = min(goal_required, amount_alloc_based)
                print(f"[DECISION] ✅ Paying Goal {goal} | ΔUtility: {net_util:.3f}")
                end_value_sum -= amount_taken
            else:
                amount_taken = 0
                print(f"[DECISION] ❌ Skipping Goal {goal} | ΔUtility: {net_util:.3f}")

            goal_payout_info[f"Goal {goal} Allocation (%)"] = alloc_pct
            goal_payout_info[f"Goal {goal} Amount Taken (DKK)"] = amount_taken
            goal_payout_info[f"Goal {goal} Utility Net Δ"] = net_util
            goal_payout_info[f"Goal {goal} Utility Loss"] = fut_loss
            goal_payout_info[f"Goal {goal} Value Now"] = value_now_nested
            goal_payout_info[f"Goal {goal} Utility φ_before"] = ", ".join(str(round(p[0], 4)) for p in phi_details)
            goal_payout_info[f"Goal {goal} Utility φ_after"] = ", ".join(str(round(p[1], 4)) for p in phi_details)

    # Annotate each row in asset_level_log for current year
    for row in asset_level_log:
        if row["Year"] == loop_year:
            for key, value in goal_payout_info.items():
                row[key] = value
    # - V9 END -

    final_wealth_tracker[loop_year] = end_value_sum

    print(f"\n--- Year {loop_year} Summary ---")
    print(f"Income (Salary): {salary:.2f}")
    if loop_year != starting_year:
        print(f"Previous Wealth Carried: {prev_wealth:.2f}")
    print(f"Post-Tax Wealth (New Final Wealth): {pool:.2f}")
    print("------------------------------\n")
#%%
# V7C: Export new asset-level log for debugging
pd.DataFrame(asset_level_log).to_csv(os.path.join(output_folder, "asset_returns_log.csv"), index=False)

#V12 moved finalwealth to python
pd.DataFrame.from_dict(final_wealth_tracker, orient='index', columns=['Final Wealth']).to_csv(
    os.path.join(output_folder, "final_wealth_by_year.csv")
)

#V15
pd.DataFrame(goal_weights_tracker).to_csv(
    os.path.join(output_folder, "goal_weights_by_year.csv"), index=False
)

pd.DataFrame(aggregate_portfolio_tracker).to_csv(
    os.path.join(output_folder, "aggregate_portfolio_by_year.csv"), index=False
)

# V15: Save optimal_weights for Goals A–D
for goal_letter, matrix_dict in zip(
    ["A", "B", "C", "D"],
    [goal_optimal_weights_A, goal_optimal_weights_B, goal_optimal_weights_C, goal_optimal_weights_D]
):
    all_dfs = []
    for year, mat in matrix_dict.items():
        df = pd.DataFrame(mat)
        df.insert(0, "Year", year)
        df.insert(1, "Allocation Level", np.arange(0.01, 1.01, 0.01))
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(output_folder, f"optimal_weights_goal_{goal_letter}.csv"), index=False)


#%%
