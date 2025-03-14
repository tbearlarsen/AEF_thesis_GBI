import os
import subprocess
from codelib.Models.vasicek_model import VasicekModel

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder = os.path.join(root, "Miscellaneous", "Archive", "Data")

    data = os.path.join(data_folder, "short_rate.xlsx")
    model = VasicekModel(data,220)
    r0=model.rates.iloc[-1]

    sim_prices_short = model.run_simulation(
        r0,
        10,
        10000,
        plot=True
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