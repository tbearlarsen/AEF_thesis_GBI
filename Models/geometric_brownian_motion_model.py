import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GBMSimulator:
    def __init__(self, prices, ppa=52):
        """
        Initialize the simulator with asset price data.

        Parameters:
            prices (pd.Series): Price series for the asset.
            ppa (str): Periods per annum ('weekly' assumed by default).
        """
        self.prices = prices
        self.ppa = ppa
        self.log_returns = self.calculate_log_returns()
        (self.mu, self.sigma,
         self.mu_annual, self.sigma_annual) = self.estimate_parameters()

    def calculate_log_returns(self):
        """Calculate log returns from the price data."""
        return np.log(self.prices / self.prices.shift(1)).dropna()

    def estimate_parameters(self):
        """
        Estimate the drift and volatility from the log returns.

        Returns:
            tuple: (mu, sigma, mu_annual, sigma_annual)
                mu: drift (weekly if data is weekly)
                sigma: volatility (weekly if data is weekly)
                mu_annual: annualized drift
                sigma_annual: annualized volatility
        """
        mu = self.log_returns.mean()
        sigma = self.log_returns.std()

        #Annualize parameters
        mu_annual = mu * self.ppa
        sigma_annual = sigma * np.sqrt(self.ppa)

        print(f"Estimated weekly drift: {mu:.4f}, Annual drift: {mu_annual:.4f}")
        print(f"Estimated weekly volatility: {sigma:.4f}, Annual volatility: {sigma_annual:.4f}")
        return mu, sigma, mu_annual, sigma_annual

    def simulate_gbm(self, num_years=45, num_paths=10000):
        """
        Simulate asset price paths using Geometric Brownian Motion (GBM).

        Parameters:
            num_years (int): Simulation horizon in years.
            num_paths (int): Number of Monte Carlo simulation paths.

        Returns:
            np.ndarray: Simulated asset prices with shape (num_steps+1, num_paths).
        """
        dt = 1 / self.ppa
        num_steps = int(self.ppa * num_years)
        S0 = self.prices.iloc[-1]

        # Vectorized simulation: generate increments for the log-price process
        increments = ((self.mu - 0.5 * self.sigma ** 2) * dt +
                      self.sigma * np.sqrt(dt) *
                      np.random.normal(0, 1, (num_steps, num_paths)))
        # Cumulative sum to obtain log-price paths
        log_price_paths = np.vstack([np.zeros(num_paths), np.cumsum(increments, axis=0)])
        simulated_prices = S0 * np.exp(log_price_paths)
        return simulated_prices

    def plot_simulation(self, simulated_prices, num_paths_to_plot=10, title="GBM Simulation"):
        """
        Plot a sample of simulated asset price paths.

        Parameters:
            simulated_prices (np.ndarray): Array of simulated asset prices.
            num_paths_to_plot (int): Number of paths to plot.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(simulated_prices[:, i], lw=1.5, alpha=0.8, label=f'Path {i + 1}')
        plt.xlabel("Time (weeks)")
        plt.ylabel("Asset Price")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    """
    Example: Use the model to simulate S&P500 index prices.
    """

    #Load the data from a module or file, and run the simulation.
    from Simulation.Data.sp500_data import close as prices

    #Check that the price data is loaded correctly
    if prices.empty:
        raise ValueError("Price data is empty or not loaded correctly.")

    #Create an instance of the simulator using the price data.
    simulator = GBMSimulator(prices,52)

    #Run the simulation for 45 years with 10,000 Monte Carlo paths.
    simulated_prices = simulator.simulate_gbm(num_years=45, num_paths=10000)

    #Plot a few sample paths
    simulator.plot_simulation(simulated_prices, num_paths_to_plot=10,
                              title="Monte Carlo Simulation of Asset Returns using GBM")