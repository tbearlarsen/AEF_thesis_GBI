import os
import subprocess
import pandas as pd
from Models.vasicek_model import VasicekModelExtended

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder = os.path.join(root, "Archive", "Data")

    data = os.path.join(data_folder, "short_rate.xlsx")
    model = VasicekModelExtended(data,220)
    r0=model.rates.iloc[-1,0]
    simulated_short_rates=model.run_model(
        r0,
        10,
        40,
        10000,
        plot=True,
        num_paths_to_plot=10
    )

    return simulated_short_rates

if __name__ == "__main__":
    simulated_short_rates=main()

"""
Denmark Short-Term Rate (DESTR) is a transaction-based reference rate based on unsecured overnight deposit transactions.
Danmarks Nationalbank calculates DESTR on all Danish banking days based on the specific transactions made in the
Danish krone market the previous banking day. Reference rates are used in a wide range of financial contracts,
including bank loans, mortgage bonds and interest rate swaps.
"""