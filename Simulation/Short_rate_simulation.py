import pandas as pd
from Models.vasicek_model import VasicekModelExtended

def main():
    data_file = r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\short_rate.xlsx"
    model = VasicekModelExtended(data_file,220)
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