from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codelib.Models.vasicek_model import VasicekModel


class ZCBYieldTimeMaturity:
    """
    A class to simulate short rates using the Vasicek model and compute zero-coupon yields.
    """

    def __init__(self, short_rate_path: str, risk_premium: float, days_per_year: int = 252, seed: int = 222,
                 sim_horizon: int = 10, years_range: range = range(1, 11), maturities_range: range = range(1, 21)):
        """
        Initialise the simulator and compute the simulated yields.

        Parameters:
            short_rate_path (str): Path to the short rate data.
            risk_premium (float): Risk premium parameter.
            days_per_year (int): Number of trading days per year (default is 252).
            seed (int): Random seed for reproducibility.
            sim_horizon (int): Simulation horizon in years.
            years_range (range): Range of simulation years (default is 1 to 10).
            maturities_range (range): Range of maturities (default is 1 to 20).
        """
        np.random.seed(seed)
        self.short_rate_path = short_rate_path
        self.risk_premium = risk_premium
        self.days_per_year = days_per_year
        self.sim_horizon = sim_horizon
        self.years_range = years_range
        self.maturities_range = maturities_range

        # Initialise the Vasicek model and run the simulation of short rates.
        self.short_rates_sim, self.kappa, self.theta, self.beta = self.run_vasicek_model()

        # Cache the simulated yields for efficiency; computed once during initialisation.
        self.simulated_yields = self.simulate_yields()

    def run_vasicek_model(self):
        model = VasicekModel(self.short_rate_path, self.days_per_year)
        kappa, theta, beta = model.estimate_params()

        r0 = model.rates.iloc[-1]
        short_rates_sim = model.run_simulation(
            r0,
            self.sim_horizon,
            10000,
            plot=False
        )
        return short_rates_sim, kappa, theta, beta

    @staticmethod
    def calculate_zero_coupon_yield(time_to_maturity: Union[float, np.ndarray],
                                    initial_short_rate: Union[float, np.ndarray],
                                    kappa: float,
                                    theta: float,
                                    beta: float,
                                    risk_premium: float) -> Union[float, np.ndarray]:
        """
        Calculate the zero-coupon yield for a given time-to-maturity.

        Parameters:
            time_to_maturity (Union[float, np.ndarray]): Time to maturity (in years).
            initial_short_rate (Union[float, np.ndarray]): The short rate at the start.
            kappa (float): Speed of mean reversion.
            theta (float): Long-term mean level.
            beta (float): Volatility parameter.
            risk_premium (float): Risk premium parameter.

        Returns:
            Union[float, np.ndarray]: The computed zero-coupon yield.
        """
        y_infty = theta - risk_premium * beta / kappa - beta ** 2 / (2 * kappa ** 2)
        b = 1 / kappa * (1 - np.exp(-kappa * time_to_maturity))
        a = y_infty * (time_to_maturity - b) + beta ** 2 / (4 * kappa) * b ** 2
        return (a + b * initial_short_rate) / time_to_maturity

    def simulate_yields(self,
                        years: range = None,
                        maturities: range = None) -> Dict[int, Dict[int, Any]]:
        """
        Compute the zero-coupon yields for multiple simulation years and maturities.

        Parameters:
            years (range): Simulation years (default is as initialised).
            maturities (range): Maturities in years (default is as initialised).

        Returns:
            Dict[int, Dict[int, Any]]: A nested dictionary where keys are simulation years and
            values are dictionaries with maturities as keys and computed yields as values.
        """
        if years is None:
            years = self.years_range
        if maturities is None:
            maturities = self.maturities_range

        simulated_yields = {}

        for year in years:
            # Compute the index corresponding to the end of the simulation year.
            index = year * self.days_per_year
            # Ensure the index does not exceed the simulation length.
            if index >= self.short_rates_sim.shape[1]:
                index = self.short_rates_sim.shape[1] - 1

            # Extract the short rates at the selected time (across all simulation paths).
            sr_at_year = self.short_rates_sim[:, index]

            simulated_yields[year] = {}
            for maturity in maturities:
                # Compute the yield for each maturity.
                yield_values = self.calculate_zero_coupon_yield(maturity,
                                                                sr_at_year,
                                                                self.kappa,
                                                                self.theta,
                                                                self.beta,
                                                                self.risk_premium)
                simulated_yields[year][maturity] = yield_values
        return simulated_yields

    def get_yields_for_year(self, year: int) -> pd.DataFrame:
        """
        Retrieve the zero-coupon yields for a specific simulation year as a pandas DataFrame.

        Parameters:
            year (int): The simulation year to inspect.

        Returns:
            pd.DataFrame: A DataFrame where each column represents a maturity and each row corresponds to a simulation path.
        """
        if year not in self.simulated_yields:
            raise ValueError(f"Year {year} is out of range for the simulated yields.")
        # Convert the dictionary for the specified year into a DataFrame.
        return pd.DataFrame(self.simulated_yields[year])

    def plot_yield_curve(self, year: int) -> None:
        """
        Plot the zero-coupon yield curve for a given simulation year.

        Parameters:
            year (int): The simulation year for which to plot the yield curve.
        """
        # Compute the index corresponding to the end of the simulation year.
        index = year * self.days_per_year
        if index >= self.short_rates_sim.shape[1]:
            index = self.short_rates_sim.shape[1] - 1

        # Extract the short rates at the specified year.
        sr_at_year = self.short_rates_sim[:, index]

        # Compute yields for each maturity.
        yields = [self.calculate_zero_coupon_yield(m, sr_at_year,
                                                   self.kappa,
                                                   self.theta,
                                                   self.beta,
                                                   self.risk_premium) for m in self.maturities_range]

        # Plot the yield curve.
        plt.figure(figsize=(8, 5))
        plt.plot(list(self.maturities_range), yields, marker='o')
        plt.title(f"Zero-Coupon Yield Curve for Simulation Year {year}")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield")
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == '__main__':
    # Replace 'XXX' with the actual path to your data.
    short_rate_path = 'XXX'
    risk_premium = 0.05  # Example risk premium

    simulator = ZCBYieldTimeMaturity(short_rate_path, risk_premium)

    # Get yields for a specified simulation year
    year_to_inspect = 5
    try:
        df_year5 = simulator.get_yields_for_year(year_to_inspect)
        print(f"Zero-coupon yields for year {year_to_inspect}:\n", df_year5.head())
    except ValueError as e:
        print(e)

    # Plot the yield curve for simulation year 5 as an example.
    simulator.plot_yield_curve(year=5)