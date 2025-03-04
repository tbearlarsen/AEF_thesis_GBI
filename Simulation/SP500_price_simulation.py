import pandas as pd
import yfinance as yf
from Models.geometric_brownian_motion_model import GBMSimulator

def main():
    """ticker = "^GSPC"
    start_date = "2010-01-01"
    end_date = "2025-01-01"
    freq = "1mo"

    data = yf.download(ticker, start=start_date, end=end_date, interval=freq)
    prices = data["Adj Close"]"""

    data = pd.read_excel(r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\sp500.xlsx")
    prices = data["Close"]

    #Check if the price data is loaded correctly
    if prices.empty:
        raise ValueError("There was a problem loading the price data.")

    #Create an instance of the simulator using the manually loaded price data
    simulator = GBMSimulator(prices)

    #Run the simulation for 45 years with 10,000 Monte Carlo paths
    simulated_prices = simulator.simulate_gbm(num_years=45, num_paths=10000)

    #Plot a few sample simulation paths
    simulator.plot_simulation(simulated_prices, num_paths_to_plot=10,
                              title="Monte Carlo Simulation of S&P500 Returns using GBM")

    return simulated_prices


if __name__ == "__main__":
    simulated_sp500_prices=main()







import numpy as np
import matplotlib.pyplot as plt

final_prices=simulated_sp500_prices[:,-1]
mean_final=np.mean(final_prices)
median_final=np.median(final_prices)
std_final=np.std(final_prices)

print("Mean final price: ", mean_final)
print("Median final price: ", median_final)
print("Standard deviation of final price: ", std_final)

#Plot the histogram of the final prices
plt.figure(figsize=(12, 6))
plt.hist(final_prices, bins=50, color="skyblue", edgecolor="black")
plt.show()

