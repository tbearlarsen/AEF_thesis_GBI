import os
from codelib.Models.vasicek_model import VasicekModel
from codelib.file_management.dynamic_file_pathing import get_root
import numpy as np

def main():
    root = get_root()
    data_folder = os.path.join(root, "Miscellaneous", "Archive", "Data")
    np.random.seed(222)

    data = os.path.join(data_folder, "short_rate.xlsx")
    model = VasicekModel(data,252)
    r0=model.rates.iloc[-1]

    sim_prices_short = model.run_simulation(
        r0,
        10,
        10000,
        plot=False
    )

    return sim_prices_short

if __name__ == "__main__":
    simulated_short_rates=main()

"""
Denmark Short-Term Rate (DESTR) is a transaction-based reference rate based on unsecured overnight deposit transactions.
Danmarks Nationalbank calculates DESTR on all Danish banking days based on the specific transactions made in the
Danish krone market the previous banking day. Reference rates are used in a wide range of financial contracts,
including bank loans, mortgage bonds and interest rate swaps.
"""


import matplotlib.pyplot as plt
from codelib.Johan.visualization.base import fan_chart

dt = 1.0 / 252
horizon = 10
num_time_steps = int(horizon / dt)
time_points = np.arange(0, num_time_steps + 1, 1) * dt

percentiles = np.percentile(simulated_short_rates, [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
fan_chart(time_points, percentiles, ax=ax, color="navy")
plt.show()