import os
import subprocess
import pandas as pd
from codelib.Models.geometric_brownian_motion_model import GBMSimulator
import matplotlib.pyplot as plt

def main():
    # Determine the root directory
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder = os.path.join(root, "Simulation", "Data")
    output_folder = os.path.join(root, "Simulation", "Nordnet", "sim_price")

    """file_directory = {
        "EUNK": "EUNK_iShares_Core_MSCI_Europe_UCITS_ETF_EUR_Acc.csv",
        "SXR8": "SXR8_iShares_Core_S&P_500_ETF_USD_Acc_EUR.csv",
        "IUSN": "IUSN_iShares_MSCI_World_Small_Cap_UCITS_ETF_USD_Acc.csv",
        "JGHY": "JGHY_JPM_Global_High_Yield_Corporate_Bond_Multi-Factor_UCITS_ETF_USD_acc.csv",
        "LYXF": "LYXF_Amundi_Euro_Government_Bond_15plusY_UCITS_ETF_Acc.csv",
        "SXR4": "SXR4_iShares_MSCI_USA_UCITS_ETF_USD_Acc.csv",
        "SYBA": "SYBA_SPDR_Bloomberg_Euro_Aggregate_Bond_UCITS_ETF_Dist.csv",
        "SYBB": "SYBB_SPDR_Bloomberg_Euro_Government_Bond_UCITS_ETF_Dist.csv",
        "SYBC": "SYBC_SPDR_Bloomberg_Euro_Corporate_Bond_UCITS_ETF_Dist.csv"
    }"""

    file_directory = {
        "EUNK": "EUNK_iShares_Core_MSCI_Europe_UCITS_ETF_EUR_Acc.csv"
    }

    # Read the historical prices
    prices = pd.DataFrame()
    for name, path in file_directory.items():
        data_path = os.path.join(data_folder, path)
        data = pd.read_csv(data_path, index_col=0, parse_dates=True, dayfirst=True)
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y")
        data = data[data.index.year < 2025]
        prices[name] = data["Slutkurs"]

    # Make a copy of the historical data for plotting purposes
    historical_data = prices.copy()

    # Loop over each asset to perform the simulation and create the combined plot
    for column in prices.columns:
        print(f"Starting simulation for {column}")
        column_data = prices[column].dropna()
        GBM = GBMSimulator(column_data, 255)
        simulated_prices = GBM.simulate_gbm(10, 100)
        simulated_prices = pd.DataFrame(simulated_prices)

        # Save the full simulated paths to Excel (optional)
        output_path = os.path.join(output_folder, f"{column}.xlsx")
        simulated_prices.to_excel(output_path, sheet_name=column, index=False, header=False)

        # Calculate the mean price across simulation paths for each simulated time step
        simulated_means = simulated_prices.mean(axis=1)

        # Create a date range starting from the day after the last historical date
        last_date = prices.index.max()
        new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(simulated_means), freq='D')

        # Extend the DataFrame's index to include the new dates
        prices = prices.reindex(prices.index.union(new_dates))

        # Now you can assign the simulated mean values to the new dates for the given asset
        prices.loc[new_dates, column] = simulated_means.values

        # Plot both historical and simulated mean prices in the same figure
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data[column], label="Historical Prices", color="blue")
        plt.plot(new_dates, simulated_means, label="Simulated Mean Prices", color="red")
        plt.title(f"{column} Prices: Historical vs Simulated Mean")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        print(f"Simulation for {column} completed\n")

if __name__ == "__main__":
    main()
