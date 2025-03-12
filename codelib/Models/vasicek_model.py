import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

class VasicekModel:
    def __init__(self, data_file, ppa):
        """
        Initialise the Vasicek model.

        Parameters:
            data_file : str
                Path to the Excel file containing the rate data.
            ppa : int
                Periods per annum: the number of observations per year.
        """
        self.data_file = data_file
        self.ppa = ppa
        self.dt = 1 / ppa
        self.rates = self.load_data()
        self.a_hat = None
        self.b_hat = None
        self.sigma_hat = None
        self.simulation_data = None

    def load_data(self):
        """Load and clean short rate data from an Excel file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        rates = pd.read_excel(self.data_file, parse_dates=[0], index_col=0)
        return rates.dropna()

    def estimate_params(self):
        """Estimate Vasicek model parameters using OLS regression."""
        # Create lagged series
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        # OLS regression: r_{t+1} = theta + phi * r_t + error
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        theta_hat = model.params[0]
        phi_hat = model.params[1]

        # Recover continuous-time parameters
        self.a_hat = -np.log(phi_hat) / self.dt
        self.b_hat = theta_hat / (1 - phi_hat)

        # Estimate volatility
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.sigma_hat = sigma_eta_hat * np.sqrt(2 * self.a_hat / (1 - phi_hat ** 2))

        print(f"Estimated a: {self.a_hat}")
        print(f"Estimated b: {self.b_hat}")
        print(f"Estimated sigma: {self.sigma_hat}")

    def simulate(self, r0, years, num_paths):
        """
        Simulate short rate paths over a specified number of years.

        Parameters:
            r0 : float
                The starting short rate.
            years : float
                Number of years for the simulation.
            num_paths : int
                Number of simulated paths.
        """
        num_periods = int(years * self.ppa)
        exp_a_dt = np.exp(-self.a_hat * self.dt)
        sigma_dt = self.sigma_hat * np.sqrt((1 - np.exp(-2 * self.a_hat * self.dt)) / (2 * self.a_hat))

        sim_rates = np.zeros((num_periods + 1, num_paths))
        sim_rates[0, :] = r0

        for t in range(1, num_periods + 1):
            eps = np.random.normal(size=num_paths)
            sim_rates[t, :] = (
                self.b_hat
                + (sim_rates[t - 1, :] - self.b_hat) * exp_a_dt
                + sigma_dt * eps
            )

        # Store the simulated paths
        self.simulation_data = sim_rates
        return sim_rates

    def plot_simulation(self, simulation, num_paths_to_plot=10,
                        xlabel="Periods", ylabel="Short Rate", title="Simulation Paths"):
        """
        Plot a sample of simulation paths.

        Parameters:
            simulation : np.ndarray
                Array of simulated short rate paths.
            num_paths_to_plot : int, optional
                Number of paths to plot (default is 10).
            xlabel : str, optional
                Label for the x-axis (default is "Periods").
            ylabel : str, optional
                Label for the y-axis (default is "Short Rate").
            title : str, optional
                Title for the plot (default is "Simulation Paths").
        """
        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(simulation[:, i], lw=1.5, label=f"Path {i + 1}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_model(self, r0, years, num_paths, plot=False, num_paths_to_plot=10):
        """
        Run the model:
          1. Estimate parameters.
          2. Simulate short rate paths for the specified number of years.
          3. Optionally plot the simulation.

        Parameters:
            r0 : float
                The starting short rate.
            years : float
                Number of years for the simulation.
            num_paths : int
                Number of simulated paths.
            plot : bool, optional
                If True, plots the simulation after running the model.
            num_paths_to_plot : int, optional
                Number of paths to plot (default is 10).

        Returns:
            np.ndarray
                The simulation data covering the specified time horizon.
        """
        self.estimate_params()
        simulation = self.simulate(r0, years, num_paths)

        if plot:
            self.plot_simulation(simulation, num_paths_to_plot=num_paths_to_plot,
                                 title="Simulation")

        return simulation


class VasicekModelExtended:
    """
    This is an extention of the Vasicek model. It simulates over a short horizon using the pure model,
    and then extends the simulation long-term using modified parameters that enhance mean reversion and volatility,
    so that the long-term simulated data is closer to the historical mean and volatility.
    """


    def __init__(self, data_file, ppa):
        """
        Initialise the Vasicek model.

        Parameters:
            data_file : str
                Path to the Excel file containing the rate data.
            ppa : int
                Periods per annum: the number of observations per year.
        """
        self.data_file = data_file
        self.ppa = ppa
        self.dt = 1 / ppa
        self.rates = self.load_data()
        self.a_hat = None
        self.b_hat = None
        self.sigma_hat = None
        self.simulation_data = None

    def load_data(self):
        """Load and clean short rate data from an Excel file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        rates = pd.read_excel(self.data_file, parse_dates=[0], index_col=0)
        return rates.dropna()

    def estimate_params(self):
        """Estimate Vasicek model parameters using OLS regression."""

        #Create lagged series
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        #OLS regression: r_{t+1} = theta + phi * r_t + error
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        theta_hat = model.params[0]
        phi_hat = model.params[1]

        #Recover continuous-time parameters
        self.a_hat = -np.log(phi_hat) / self.dt
        self.b_hat = theta_hat / (1 - phi_hat)

        #Estimate volatility
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.sigma_hat = sigma_eta_hat * np.sqrt(2 * self.a_hat / (1 - phi_hat ** 2))

        print(f"Estimated a: {self.a_hat}")
        print(f"Estimated b: {self.b_hat}")
        print(f"Estimated sigma: {self.sigma_hat}")

    def simulate(self, r0, years, num_paths):
        """
        Simulate short rate paths over a specified number of years.

        Parameters:
            r0 : float
                The starting short rate.
            years : float
                Number of years for the simulation.
            num_paths : int
                Number of simulated paths.
        """
        num_periods = int(years * self.ppa)
        exp_a_dt = np.exp(-self.a_hat * self.dt)
        sigma_dt = self.sigma_hat * np.sqrt(
            (1 - np.exp(-2 * self.a_hat * self.dt)) / (2 * self.a_hat)
        )

        sim_rates = np.zeros((num_periods + 1, num_paths))
        sim_rates[0, :] = r0

        for t in range(1, num_periods + 1):
            eps = np.random.normal(size=num_paths)
            sim_rates[t, :] = (
                    self.b_hat
                    + (sim_rates[t - 1, :] - self.b_hat) * exp_a_dt
                    + sigma_dt * eps
            )

        #Store the simulated paths
        self.simulation_data = sim_rates
        return sim_rates

    def extend_simulation(self, total_years, k=3, c=0.5):
        """
        Extend the simulation with enhanced mean reversion parameters.

        Parameters:
            total_years : float
                Total number of years for the extended simulation.
            k : float, optional
                Factor to adjust the mean reversion speed in the extended simulation (default is 3).
            c : float, optional
                Factor to adjust the volatility in the extended simulation (default is 0.5).

        Note: This method uses the simulation stored in self.simulation_data as the starting point.
        """
        if self.simulation_data is None:
            raise ValueError("No simulation available. Please run simulate() first.")

        num_periods_sim = self.simulation_data.shape[0] - 1
        num_paths = self.simulation_data.shape[1]
        total_periods = int(total_years * self.ppa)
        extended_rates = np.zeros((total_periods + 1, num_paths))

        #Import the paths from the initial simulation
        extended_rates[: num_periods_sim + 1, :] = self.simulation_data

        #Modified parameters for the extended simulation.
        a_long = self.a_hat * k
        sigma_long = self.sigma_hat * c

        exp_a_dt_long = np.exp(-a_long * self.dt)
        sigma_dt_long = sigma_long * np.sqrt(
            (1 - np.exp(-2 * a_long * self.dt)) / (2 * a_long)
        )

        for t in range(num_periods_sim + 1, total_periods + 1):
            eps = np.random.normal(size=num_paths)
            extended_rates[t, :] = (
                    self.b_hat
                    + (extended_rates[t - 1, :] - self.b_hat) * exp_a_dt_long
                    + sigma_dt_long * eps
            )

        #Update the simulation_data variable with the full simulation.
        self.simulation_data = extended_rates
        return extended_rates

    def plot_simulation(self,
            simulation,
            num_paths_to_plot=10,
            xlabel="Periods",
            ylabel="Short Rate",
            title="Simulation Paths",
    ):
        """
        Plot a sample of simulation paths.

        Parameters:
            simulation : np.ndarray
                Array of simulated short rate paths.
            num_paths_to_plot : int, optional
                Number of paths to plot (default is 10).
            xlabel : str, optional
                Label for the x-axis (default is "Periods").
            ylabel : str, optional
                Label for the y-axis (default is "Short Rate").
            title : str, optional
                Title for the plot (default is "Simulation Paths").
        """
        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(simulation[:, i], lw=1.5, label=f"Path {i + 1}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_model(self, r0, initial_years, total_years, num_paths, k=3, c=0.5,
                  plot=False, num_paths_to_plot=10):
        """
        Run the entire model in one call.

        This method performs the following steps:
          1. Estimates model parameters.
          2. Simulates the short rate for a specified number of initial years.
          3. Extends the simulation to cover the total number of years.
          4. Optionally plots the full simulation.

        Parameters:
        r0 : float
            The starting short rate.
        initial_years : float
            Number of years for the initial simulation.
        total_years : float
            Total number of years for the extended simulation.
        num_paths : int
            Number of simulated paths.
        k : float, optional
            Factor to adjust the mean reversion speed in the extended simulation (default is 3).
        c : float, optional
            Factor to adjust the volatility in the extended simulation (default is 0.5).
        plot : bool, optional
            If True, plots the full simulation after running the model.
        num_paths_to_plot : int, optional
            Number of paths to plot (default is 10).

        Returns:
        np.ndarray
            The complete simulation data covering the full time horizon.
        """
        self.estimate_params()
        self.simulate(r0, initial_years, num_paths)
        self.extend_simulation(total_years, k, c)

        if plot:
            self.plot_simulation(self.simulation_data, num_paths_to_plot=num_paths_to_plot,
                                 title="Full Simulation")
        return self.simulation_data


"""
Example use of the extended model
"""
if __name__ == "__main__":
    data_file = r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\short_rate.xlsx"

    #Initialise the model
    model = VasicekModelExtended(data_file,220)

    #Define r0:
    r0=model.rates.iloc[-1,0]

    #Run the model
    simulated_short_rates=model.run_model(
        r0,
        10,
        40,
        10000,
        3,
        0.5,
        plot=True,
        num_paths_to_plot=10
    )


"""
Example use of the model
"""
if __name__ == "__main__":
    data_file = r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\short_rate.xlsx"

    #Initialise the model
    model = VasicekModel(data_file, 220)

    #Define r0:
    r0 = model.rates.iloc[-1, 0]

    #Run the model
    simulated_short_rates = model.run_model(
        r0,
        10,
        10000,
        plot=True,
        num_paths_to_plot=10
    )
