import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os


class VasicekModel:
    def __init__(self, data_file, days_per_annum=220):
        self.data_file = data_file
        self.days_per_annum = days_per_annum
        self.dt = 1 / days_per_annum
        self.rates = self.load_data()
        self.a_hat = None
        self.b_hat = None
        self.sigma_hat = None

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
        """Simulate short rate paths over a specified number of years."""
        num_days = int(years * self.days_per_annum)
        exp_a_dt = np.exp(-self.a_hat * self.dt)
        sigma_dt = self.sigma_hat * np.sqrt((1 - np.exp(-2 * self.a_hat * self.dt)) / (2 * self.a_hat))

        sim_rates = np.zeros((num_days + 1, num_paths))
        sim_rates[0, :] = r0

        for t in range(1, num_days + 1):
            eps = np.random.normal(size=num_paths)
            sim_rates[t, :] = (self.b_hat +
                               (sim_rates[t - 1, :] - self.b_hat) * exp_a_dt +
                               sigma_dt * eps)
        return sim_rates

    def extend_simulation(self, simulated_rates, total_years, k=3, c=0.5):
        """Extend the simulation with enhanced mean reversion parameters."""
        num_days_sim = simulated_rates.shape[0] - 1
        num_paths = simulated_rates.shape[1]
        total_days = int(total_years * self.days_per_annum)
        extended_rates = np.zeros((total_days + 1, num_paths))
        extended_rates[:num_days_sim + 1, :] = simulated_rates

        # Modified parameters
        a_long = self.a_hat * k
        sigma_long = self.sigma_hat * c

        exp_a_dt_long = np.exp(-a_long * self.dt)
        sigma_dt_long = sigma_long * np.sqrt((1 - np.exp(-2 * a_long * self.dt)) / (2 * a_long))

        for t in range(num_days_sim + 1, total_days + 1):
            eps = np.random.normal(size=num_paths)
            extended_rates[t, :] = (self.b_hat +
                                    (extended_rates[t - 1, :] - self.b_hat) * exp_a_dt_long +
                                    sigma_dt_long * eps)
        return extended_rates

    def plot_simulation(self, simulation, num_paths_to_plot=10, xlabel="Days", ylabel="Short Rate",
                        title="Simulation Paths"):
        """Plot a sample of simulation paths."""
        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(simulation[:, i], lw=1.5, label=f'Path {i + 1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    #data_file = r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/short_rate.xlsx"
    data_file = r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\short_rate.xlsx"
    model = VasicekModel(data_file)
    model.estimate_params()

    # Initial rate: last observed rate in the data
    r0 = model.rates.iloc[-1, 0]

    # Simulate for 10 years and plot
    sim_rates = model.simulate(r0, years=10, num_paths=10000)
    model.plot_simulation(sim_rates, num_paths_to_plot=10, title="10-Year Short Rate Simulation")

    # Extend simulation to 40 years with enhanced mean reversion
    extended_rates = model.extend_simulation(sim_rates, total_years=40, k=3, c=0.5)
    model.plot_simulation(extended_rates, num_paths_to_plot=10, title="40-Year Extended Short Rate Simulation")
