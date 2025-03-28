import os
from codelib.file_management.dynamic_file_pathing import get_root
import pandas as pd
import numpy as np
from codelib.Models.vasicek_model import VasicekRiskNeutralEstimator
import matplotlib.pyplot as plt

# Get the data file path
root = get_root()
data_folder = os.path.join(root, 'Simulation', 'Bonds', 'Data')
data_file = os.path.join(data_folder, 'monthly_rates.xlsx')

# Parameters: periods per annum and risk premium
ppa = 12    #monthly data
risk_premium = -0.05

# Create an instance of the estimator (it will load data internally)
model = VasicekRiskNeutralEstimator(data_file, ppa, risk_premium)

# Estimate the Vasicek parameters and compute risk-neutral measure
model.estimate_params()
model.compute_risk_neutral()

# Use the last available short rate for pricing
r_current = model.rates.iloc[-1]

# Define maturities (in years) for zero-coupon bonds
maturities = [1, 5, 10, 20, 30]
model_yields = {}

print("\nModel-Implied Yields:")
for T in maturities:
    price = model.price_zcb(r_current, T)
    yield_cc = model.yield_from_price(price, T)
    model_yields[T] = yield_cc
    print(f"  {T}Y yield: {yield_cc:.4%}")

# Extract observed yields using the correct column names
try:
    observed_yields = model.data[['1Y', '5Y', '10Y', '20Y', '30Y']]
    observed_yields_last = observed_yields.iloc[-1].to_dict()

    print("\nObserved Yields (last observation):")
    for T in maturities:
        key = f"{T}Y"  # Use the correct key based on your Excel column names
        print(f"  {T}Y yield: {observed_yields_last.get(key, np.nan):.4%}")

    # Prepare lists for plotting
    model_yields_list = [model_yields[T] for T in maturities]
    observed_yields_list = [observed_yields_last.get(f"{T}Y", np.nan) for T in maturities]

    # Plot model-implied vs observed yields
    plt.figure(figsize=(8, 5))
    plt.plot(maturities, model_yields_list, label='Model-Implied Yields', marker='o')
    plt.plot(maturities, observed_yields_list, label='Observed Yields', marker='x')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Continuously Compounded Yield')
    plt.title('Comparison of Model-Implied and Observed Yields')
    plt.legend()
    plt.grid(True)
    plt.show()

except KeyError:
    print("Observed yields not found in data; skipping yield comparison plot.")
