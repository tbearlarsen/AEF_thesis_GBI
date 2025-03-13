from Models.geometric_brownian_motion_model import GBMSimulator
import pandas as pd


def main():
    data=pd.read_csv(r"/Simulation/Data/EUNK_iShares_Core_MSCI_Europe_UCITS_ETF_EUR_Acc.csv", index_col=0, parse_dates=True)
    prices=data['Slutkurs']

    GBM = GBMSimulator(prices,255)
    simulated_prices = GBM.simulate_gbm(45,1000)
    GBM.plot_simulation(simulated_prices,10,"GBM Simulation of EUNK Prices")

    return simulated_prices

if __name__ == "__main__":
    results = main()


import matplotlib.pyplot as plt
last_year=results[-1,:]
plt.figure(figsize=(10, 6))
plt.hist(last_year, bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Simulated Prices for the Last Year')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
