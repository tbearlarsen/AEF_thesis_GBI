import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation.Data.sp500_data import close as prices


class SP500Simulation:
    def __init__(self, prices, time_unit='weekly'):
        """
        Initialize the simulator with price data.

        Parameters:
            prices (pd.Series): Price series for the S&P500.
            time_unit (str): Frequency of the data ('weekly' assumed).
        """
        self.prices = prices
        self.time_unit = time_unit
        self.log_returns = self.calculate_log_returns()
        (self.mu_weekly,
         self.sigma_weekly,
         self.mu_annual,
         self.sigma_annual) = self.estimate_parameters()

    def calculate_log_returns(self):
        """Calculate log returns from the price data."""
        return np.log(self.prices / self.prices.shift(1)).dropna()

    def estimate_parameters(self):
        """
        Estimate the weekly drift and volatility from the log returns.

        Returns:
            tuple: (mu_weekly, sigma_weekly, mu_annual, sigma_annual)
        """
        mu_weekly = self.log_returns.mean()
        sigma_weekly = self.log_returns.std()
        mu_annual = mu_weekly * 52
        sigma_annual = sigma_weekly * np.sqrt(52)

        print(f"Estimated weekly drift: {mu_weekly:.4f}, Annual drift: {mu_annual:.4f}")
        print(f"Estimated weekly volatility: {sigma_weekly:.4f}, Annual volatility: {sigma_annual:.4f}")
        return mu_weekly, sigma_weekly, mu_annual, sigma_annual

    def simulate_gbm(self, num_years=45, num_paths=10000):
        """
        Simulate asset price paths using Geometric Brownian Motion (GBM).

        Parameters:
            num_years (int): Simulation horizon in years.
            num_paths (int): Number of Monte Carlo simulation paths.

        Returns:
            np.ndarray: Simulated asset prices with shape (num_steps+1, num_paths).
        """
        dt = 1 / 52  # Weekly time step
        num_steps = int(52 * num_years)
        S0 = self.prices.iloc[-1]

        # Vectorized simulation: generate increments for the log-price process
        increments = ((self.mu_weekly - 0.5 * self.sigma_weekly ** 2) * dt +
                      self.sigma_weekly * np.sqrt(dt) *
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
    # Ensure that the price data is loaded properly
    if prices.empty:
        raise ValueError("Price data is empty or not loaded correctly.")

    # Create an instance of the simulator using the imported price data
    simulator = SP500Simulation(prices)

    # Run the simulation for 45 years with 10,000 Monte Carlo paths
    simulated_prices = simulator.simulate_gbm(num_years=45, num_paths=10000)

    # Plot a few sample paths
    simulator.plot_simulation(simulated_prices, num_paths_to_plot=10,
                              title="Monte Carlo Simulation of S&P500 Returns using GBM")
