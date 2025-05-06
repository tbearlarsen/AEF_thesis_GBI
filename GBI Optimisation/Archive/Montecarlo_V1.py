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
import xlwings as xw
import time

#%%
#Case Study Profile Selection
Profile = "P1" #Either P1 or P2

#Excel worksheets - Master
excel_returns = "Returns"
excel_volatilities = "Volatilities"
excel_correlation = "Correlation"
excel_gbi = "GBI Allocations P1" if Profile == "P1" else "GBI Allocations P2"
excel_gbi_goals = "GBI Goals P1" if Profile == "P1" else "GBI Goals P2"
excel_final_wealth = "FinalWealth"
excel_income = "Salary"

#Excel worksheets - Montecarlo Returns
excel_obligations_zc = "Government ZC Bonds - Obli"
excel_obligations_igb = "Investment Grade Bonds - Obli"
excel_obligations_hyb = "High Yield Bonds - Obli"
excel_obligations_ems = "Emerging Markets State - Obli"
excel_equities_dm = "Developed Markets - Equities"
excel_equities_em = "Emerging Markets - Equities"
#%%

## -- Repo Root and Folders -- ##

# Get repo root and set folders
root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
data_folder = os.path.join(root, "GBI Optimisation", "data")
output_folder = os.path.join(root, "GBI Optimisation")

# Get excel file and sheets
master_excel_path = os.path.join(data_folder, "Master.xlsx")
montecarlo_excel_path = os.path.join(data_folder, "Montecarlo Returns.xlsx")

#Master Excel File DFs
df_vols = pd.read_excel(master_excel_path, sheet_name=excel_volatilities)

df_vols.set_index(df_vols.columns[0], inplace=True)

df_corr = pd.read_excel(master_excel_path, sheet_name=excel_correlation)
df_corr.set_index(df_corr.columns[0], inplace=True)

#Montecarlo Excel File DFs
df_zc = pd.read_excel(montecarlo_excel_path, sheet_name=excel_obligations_zc)
df_igb = pd.read_excel(montecarlo_excel_path, sheet_name=excel_obligations_igb)
df_hyb = pd.read_excel(montecarlo_excel_path, sheet_name=excel_obligations_hyb)
df_ems = pd.read_excel(montecarlo_excel_path, sheet_name=excel_obligations_ems)
df_dm = pd.read_excel(montecarlo_excel_path, sheet_name=excel_equities_dm)
df_em = pd.read_excel(montecarlo_excel_path, sheet_name=excel_equities_em)




#%%
sheet_map = {
    "Government ZC Bonds - Obligation": df_zc,
    "Investment Grade Bonds - Obligation": df_igb,
    "High Yield Bonds - Obligation": df_hyb,
    "Emerging Markets State - Obligation": df_ems,
    "Developed Markets - Equities": df_dm,
    "Emerging Markets - Equities": df_em,
}

# Stack into a long format with asset label and path number
all_paths = []

for sheet_name, df in sheet_map.items():
    df = df.drop(columns=df.columns[0])  # Drop the unnamed column
    df["Asset"] = sheet_name
    df["Path"] = range(len(df))
    all_paths.append(df)

df_returns = pd.concat(all_paths, ignore_index=True)
df_returns = df_returns.set_index(["Asset", "Path"])  # Multi-index for looping per asset/path

#%%

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
