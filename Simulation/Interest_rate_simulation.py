import pandas as pd
from Models.vasicek_model import VasicekModel

def main():
    data_file= r"/Simulation/Data/10YBond.xlsx"
    model=VasicekModel(data_file,52)
    r0=model.rates.iloc[-1,0]
    simulated_interest_rates=model.run_model(
        r0,
        45,
        10000,
        plot=True,
        num_paths_to_plot=10
    )

    return simulated_interest_rates

if __name__ == "__main__":
    interest_rates=main()