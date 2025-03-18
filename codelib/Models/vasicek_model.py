import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

class VasicekModel:
    def __init__(self, data_file, ppa):
        """
        Initializes the Vasicek model class.

        Parameters:
            data_file : str
                Path to the interest rate data file.
            ppa : int
                Periods per annum (e.g., 52 for weekly data).
        """
        self.ppa = ppa
        self.dt = 1 / ppa
        self.data_file = data_file
        self.rates = self.load_data()

        #Parameters initialized as None; will be estimated
        self.kappa = None  # Mean-reversion speed
        self.theta = None  # Long-term mean
        self.beta = None  # Volatility
        self.simulation_data = None

    def load_data(self):
        """Load interest rate data from file."""
        rates = pd.read_excel(self.data_file, parse_dates=True, index_col=0).iloc[:, 0]
        return rates

    def estimate_params(self):
        """Estimate Vasicek parameters via OLS regression."""
        #Prepare lagged data for regression
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        #Perform OLS regression (r_{t+1} = theta + phi * r_t + error)
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        phi_hat = model.params[1]
        theta_hat = model.params[0]

        #Convert discrete parameters into continuous-time Vasicek parameters
        self.kappa = -np.log(phi_hat) / self.dt
        self.theta = theta_hat / (1 - phi_hat)

        #Estimate volatility parameter
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.beta = sigma_eta_hat * np.sqrt(2 * self.kappa / (1 - phi_hat ** 2))

        #Display estimated parameters clearly
        print(f"Estimated kappa (mean reversion speed): {self.kappa:.4f}")
        print(f"Estimated theta (long-term mean): {self.theta:.4f}")
        print(f"Estimated beta (volatility): {self.beta:.4f}")

        return self.kappa, self.theta, self.beta

    def simulate(self, initial_short_rate: float, years: float, num_sim: int, z_mat=None):
        """
        Simulate short-rate paths using the Vasicek model.

        Parameters:
            initial_short_rate : float
                The initial short rate.
            years : float
                Simulation horizon in years.
            num_sim : int
                Number of simulation paths.
            z_mat : np.ndarray, optional
                External matrix of standard normal random shocks.
                Should have dimensions (num_sim, num_periods).
                If None, shocks are generated internally.
        """
        if any(param is None for param in [self.kappa, self.theta, self.beta]):
            raise ValueError("Parameters kappa, theta, and beta must be estimated first. Call estimate_params().")

        num_periods = int(years * self.ppa)
        exp_kappa_dt = np.exp(-self.kappa * self.dt)
        std_rates = self.beta * np.sqrt((1 - np.exp(-2 * self.kappa * self.dt)) / (2 * self.kappa))

        sim_rates = np.zeros((num_sim, num_periods + 1))
        sim_rates[:, 0] = initial_short_rate

        # Generate or use provided external random shocks
        if z_mat is None:
            eps = np.random.normal(size=(num_sim, num_periods))
        else:
            if z_mat.shape != (num_sim, num_periods):
                raise ValueError(f"z_mat should be of shape ({num_sim}, {num_periods})")
            eps = z_mat

        for t in range(1, num_periods + 1):
            sim_rates[:, t] = (
                    self.theta
                    + (sim_rates[:, t - 1] - self.theta) * exp_kappa_dt
                    + std_rates * eps[:, t - 1]
            )

        self.simulation_data = sim_rates
        return sim_rates

    def plot_simulation(self, sim_data, num_paths_to_plot=10, title="Simulated Vasicek Short Rate Paths"):
        """
        Plot simulated short rate paths.

        Parameters:
            sim_data : np.ndarray
                Simulated short-rate data array.
            num_paths_to_plot : int, optional
                Number of simulated paths to plot.
            title : str, optional
                Plot title.
        """
        plt.figure(figsize=(10, 6))
        for i in range(num_paths_to_plot):
            plt.plot(np.arange(sim_data.shape[1]) * self.dt, sim_data[i], label=f'Path {i + 1}', lw=1.5)

        plt.xlabel('Time (Years)')
        plt.ylabel('Short Rate')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_simulation(self, initial_short_rate, years=10, num_sim=10000, plot=True, num_paths_to_plot=10):
        """
        Estimate parameters, run simulations, and plot.

        Parameters:
            initial_short_rate : float
                Starting value for the short rate.
            years : float
                Simulation horizon (in years).
            num_sim : int
                Number of simulated paths.
            num_paths_to_plot : int, optional
                Number of simulated paths to plot if plotting is enabled.
            plot : bool, optional
                Whether to plot results after simulation.
        """
        self.estimate_params()
        sim_data = self.simulate(initial_short_rate, years, num_sim)

        if plot:
            self.plot_simulation(sim_data, num_paths_to_plot=num_paths_to_plot)

        return sim_data





