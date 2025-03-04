import matplotlib.pyplot as plt
import pandas as pd
from Models.hybrid_garch_ml import garch_ml_model



def main():
    # Define parameters for your ETF data
    filepath = r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\iShares_Core_S&P_500_ETF_USD_Acc_EUR.csv"
    date_column = "Dato"
    price_column = "Slutkurs"

    # Create an instance of ETFForecaster
    forecaster = garch_ml_model(filepath, date_column, price_column)

    # Run a simple forecast (one simulation path over the full horizon),
    # export the price path to Excel, show model details, and plot the path.
    simple_paths = forecaster.run_forecast(
        mode="simple",
        horizon=2520,
        export=True,
        filename="simple_forecast.xlsx",
        work_dir=".",  # Export to the current directory
        show_details=True,
        plot_paths=True
    )
    print("Simple simulation path exported.")



if __name__ == "__main__":
    main()
