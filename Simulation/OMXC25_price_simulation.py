import pandas as pd
import yfinance as yf
from Models.geometric_brownian_motion_model import GBMSimulator

def main():
    ticker = "^OMXC25"
    start_date = "2017-01-01"
    end_date = "2025-01-01"
    freq = "1mo"

    data = yf.download(ticker, start=start_date, end=end_date, interval=freq)
    prices = data["Adj Close"]

    #Check if the price data is loaded correctly
    if prices.empty:
        raise ValueError("The 'close' column in the Excel file is empty or missing.")

    #Create an instance of the simulator using the manually loaded price data
    simulator = GBMSimulator(prices)

    #Run the simulation for 45 years with 10,000 Monte Carlo paths
    simulated_prices = simulator.simulate_gbm(num_years=45, num_paths=10000)

    #Plot a few sample simulation paths
    simulator.plot_simulation(simulated_prices, num_paths_to_plot=10,
                              title="Monte Carlo Simulation of S&P500 Returns using GBM")

    return simulated_prices


if __name__ == "__main__":
    simulated_omxc25_prices=main()