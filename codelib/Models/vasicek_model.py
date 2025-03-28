import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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


class VasicekEstimator:
    def __init__(self, data_file, ppa):
        """
        Initializes the Vasicek model class.

        Parameters:
            data_file : str
                Path to the interest rate data file.
            ppa : int
                Periods per annum (e.g., 12 for monthly data).
        """
        self.ppa = ppa
        self.dt = 1 / ppa
        self.data_file = data_file
        self.data = self.load_data()  # load full dataframe
        self.rates = self.data["Short Rate"]  # use "Short Rate" column

        # Parameters (to be estimated)
        self.kappa = None  # Real-world mean-reversion speed
        self.theta = None  # Real-world long-term mean
        self.beta = None  # Volatility (sigma)

    def load_data(self):
        """
        Load interest rate data from file.

        Expects the Excel file to have columns:
            "Short Rate", "1Y", "5Y", "10Y", "20Y", "30Y"
        """
        data = pd.read_excel(self.data_file, parse_dates=True, index_col=0)
        return data

    def estimate_params(self):
        """
        Estimate Vasicek parameters via OLS regression on the short rate.
        The regression is of the form:
            r_{t+1} = theta_hat + phi_hat * r_t + error
        which is then transformed to obtain the continuous-time parameters.
        """
        # Prepare lagged data for regression
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        # Perform OLS regression (r_{t+1} = theta_hat + phi_hat * r_t + error)
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        phi_hat = model.params[1]
        theta_hat = model.params[0]

        # Convert discrete parameters into continuous-time Vasicek parameters
        self.kappa = -np.log(phi_hat) / self.dt
        self.theta = theta_hat / (1 - phi_hat)

        # Estimate volatility parameter
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.beta = sigma_eta_hat * np.sqrt(2 * self.kappa / (1 - phi_hat ** 2))

        # Display estimated parameters
        print(f"Estimated kappa (mean reversion speed): {self.kappa:.4f}")
        print(f"Estimated theta (long-term mean): {self.theta:.4f}")
        print(f"Estimated beta (volatility): {self.beta:.4f}")

        return self.kappa, self.theta, self.beta

    @staticmethod
    def vasicek_bond_price(r, kappa_q, theta_q, sigma, tau):
        """
        Computes the zero-coupon bond price under the Vasicek risk-neutral dynamics.

        Parameters:
            r : float
                The current short rate.
            kappa_q : float
                Risk-neutral mean reversion speed.
            theta_q : float
                Risk-neutral long-term mean.
            sigma : float
                Volatility.
            tau : float
                Time to maturity in years.

        Returns:
            Price : float
                The model-implied bond price.
        """
        B = (1 - np.exp(-kappa_q * tau)) / kappa_q
        A = np.exp((theta_q - sigma ** 2 / (2 * kappa_q ** 2)) * (B - tau) - (sigma ** 2 * B ** 2) / (4 * kappa_q))
        return A * np.exp(-B * r)

    def estimate_risk_premium(self):
        """
        Estimate the risk premium (lambda) by calibrating the risk-neutral parameters
        (kappa_q and theta_q) to the observed bond yields.

        The method uses the following columns in self.data:
            "Short Rate" for r,
            "1Y", "5Y", "10Y", "20Y", "30Y" for yields.

        Observed yields (assumed continuously compounded) are converted into discount bond prices via:
            Price = exp(-yield * maturity)

        The objective function minimizes the sum of squared errors between observed bond prices
        and the Vasicek model-implied prices over all observations and maturities.

        Returns:
            lambda_est : float
                The estimated risk premium, computed as (kappa_q - kappa).
        """
        # Ensure real-world parameters are already estimated
        if self.kappa is None or self.theta is None or self.beta is None:
            raise ValueError("Real-world parameters must be estimated first. Run estimate_params().")

        data = self.data.copy()
        r = data["Short Rate"].values  # short rate time series

        # Define the maturities (in years) for which we have yield data
        maturities = np.array([1, 5, 10, 20, 30])
        # Create a dictionary mapping maturity to observed discount bond prices
        observed_prices = {}
        for m in maturities:
            col = f"{m}Y"
            yield_values = data[col].values
            # Convert continuously compounded yields to bond prices: P = exp(-yield * m)
            observed_prices[m] = np.exp(-yield_values * m)

        # Define the objective function: sum of squared errors across all times and maturities
        def objective(params):
            kappa_q, theta_q = params
            error = 0.0
            for i in range(len(r)):
                r_i = r[i]
                for m in maturities:
                    P_model = VasicekEstimator.vasicek_bond_price(r_i, kappa_q, theta_q, self.beta, m)
                    error += (P_model - observed_prices[m][i]) ** 2
            return error

        # Use an initial guess based on the real-world parameters
        initial_guess = [self.kappa, self.theta]
        bounds = [(1e-4, 5.0), (1e-4, 0.2)]
        res = minimize(objective, initial_guess, bounds=bounds)
        kappa_q, theta_q = res.x

        # The risk premium is given by lambda = kappa_q - kappa
        lambda_est = kappa_q - self.kappa
        print(f"Estimated risk-neutral kappa: {kappa_q:.4f}, theta: {theta_q:.4f}")
        print(f"Estimated risk premium (lambda): {lambda_est:.4f}")
        return lambda_est


class VasicekRiskNeutralEstimator:
    def __init__(self, data_file, ppa, risk_premium):
        """
        Initialize the Vasicek model.

        Parameters:
          data_file (str): Path to the Excel file containing the data.
          ppa (int or float): Periods per annum (e.g. 12 for monthly data).
          risk_premium (float): Fixed market price of risk (λ).
        """
        self.data_file = data_file
        self.ppa = ppa
        self.dt = 1 / ppa
        self.risk_premium = risk_premium  # λ (market price of risk)

        # Load the data and extract the short rate
        self.data = self.load_data()
        self.rates = self.load_short_rate()

        # Parameters to be estimated (initialised as None)
        self.kappa = None   # Mean reversion speed
        self.theta = None   # Long-term mean (physical measure)
        self.beta = None    # Volatility
        self.theta_Q = None # Risk-neutral long-term mean

    def load_data(self):
        """
        Load the interest rate data from the Excel file.

        Returns:
          DataFrame: Pandas DataFrame with the data.
        """
        # Parse dates and assume the index is the date column.
        data = pd.read_excel(self.data_file, parse_dates=True, index_col=0)
        return data

    def load_short_rate(self):
        """
        Extract the short rate from the loaded data.

        Returns:
          Series: Pandas Series containing the short rate.
        """
        # Assumes the first column of the Excel file contains the short rate.
        rates = pd.to_numeric(self.data.iloc[:, 0], errors='coerce').dropna()
        return rates

    def estimate_params(self):
        """
        Estimate the Vasicek parameters (κ, θ, β) from the short rate time series.

        Uses an OLS regression of r(t+1) on r(t) and converts discrete-time estimates
        to continuous-time parameters.

        Returns:
          tuple: (kappa, theta, beta)
        """
        # Prepare lagged data
        r_t = self.rates[:-1].values.astype(float)
        r_t1 = self.rates[1:].values.astype(float)

        # Perform OLS regression: r(t+1) = theta + phi * r(t) + error
        X = sm.add_constant(r_t)
        reg_model = sm.OLS(r_t1, X).fit()
        print(reg_model.summary())

        phi_hat = reg_model.params[1]
        theta_hat = reg_model.params[0]

        # Convert to continuous-time parameters
        self.kappa = -np.log(phi_hat) / self.dt
        self.theta = theta_hat / (1 - phi_hat)

        # Estimate volatility from residuals
        sigma_eta_hat = np.std(reg_model.resid, ddof=1)
        self.beta = sigma_eta_hat * np.sqrt(2 * self.kappa / (1 - phi_hat ** 2))

        print(f"Estimated kappa (mean reversion speed): {self.kappa:.4f}")
        print(f"Estimated theta (long-term mean): {self.theta:.4f}")
        print(f"Estimated beta (volatility): {self.beta:.4f}")
        return self.kappa, self.theta, self.beta

    def compute_risk_neutral(self):
        """
        Compute the risk-neutral long-term mean using:

          theta_Q = theta - (risk_premium * beta) / kappa.

        Returns:
          float: The risk-neutral long-term mean.
        """
        self.theta_Q = self.theta - (self.risk_premium * self.beta) / self.kappa
        print(f"Risk-neutral long-term mean (theta_Q): {self.theta_Q:.4f}")
        return self.theta_Q

    def price_zcb(self, r_current, T):
        """
        Price a zero-coupon bond with maturity T (in years) using the Vasicek model.

        Parameters:
          r_current (float): The current short rate.
          T (float): Time to maturity in years.

        Returns:
          float: The price of the zero-coupon bond.
        """
        # Calculate B(t,T)
        B = (1 - np.exp(-self.kappa * T)) / self.kappa

        # Calculate A(t,T) using risk-neutral parameters
        A = np.exp((self.theta_Q - (self.beta ** 2) / (2 * self.kappa ** 2)) * (B - T)
                   - (self.beta ** 2 * B ** 2) / (4 * self.kappa))

        # Compute bond price
        price = A * np.exp(-B * r_current)
        return price

    def yield_from_price(self, price, T):
        """
        Convert the zero-coupon bond price into a continuously compounded yield.

        Parameters:
          price (float): The bond price.
          T (float): Time to maturity in years.

        Returns:
          float: The continuously compounded yield.
        """
        yield_cc = -np.log(price) / T
        return yield_cc


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

class VasicekEstimatorSimple:
    def __init__(self, data_file, ppa):
        """
        Initializes the Vasicek model class.

        Parameters:
            data_file : str
                Path to the interest rate data file.
            ppa : int
                Periods per annum (e.g., 12 for monthly data).
        """
        self.ppa = ppa
        self.dt = 1 / ppa
        self.data_file = data_file
        self.data = self.load_data()  # load full dataframe
        self.rates = self.data["Short Rate"]  # use "Short Rate" column

        # Parameters (to be estimated)
        self.kappa = None  # Real-world mean-reversion speed
        self.theta = None  # Real-world long-term mean
        self.beta = None   # Volatility (sigma)

    def load_data(self):
        """
        Load interest rate data from file.

        Expects the Excel file to have columns:
            "Short Rate", "10Y"
        """
        data = pd.read_excel(self.data_file, parse_dates=True, index_col=0)
        return data

    def estimate_params(self):
        """
        Estimate Vasicek parameters via OLS regression on the short rate.
        The regression is of the form:
            r_{t+1} = theta_hat + phi_hat * r_t + error
        which is then transformed to obtain the continuous-time parameters.
        """
        # Prepare lagged data for regression
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        # Perform OLS regression (r_{t+1} = theta_hat + phi_hat * r_t + error)
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        phi_hat = model.params[1]
        theta_hat = model.params[0]

        # Convert discrete parameters into continuous-time Vasicek parameters
        self.kappa = -np.log(phi_hat) / self.dt
        self.theta = theta_hat / (1 - phi_hat)

        # Estimate volatility parameter
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.beta = sigma_eta_hat * np.sqrt(2 * self.kappa / (1 - phi_hat ** 2))

        # Display estimated parameters
        print(f"Estimated kappa (mean reversion speed): {self.kappa:.4f}")
        print(f"Estimated theta (long-term mean): {self.theta:.4f}")
        print(f"Estimated beta (volatility): {self.beta:.4f}")

        return self.kappa, self.theta, self.beta

    @staticmethod
    def vasicek_bond_price(r, kappa_q, theta_q, sigma, tau):
        """
        Computes the zero-coupon bond price under the Vasicek risk-neutral dynamics.

        Parameters:
            r : float
                The current short rate.
            kappa_q : float
                Risk-neutral mean reversion speed.
            theta_q : float
                Risk-neutral long-term mean.
            sigma : float
                Volatility.
            tau : float
                Time to maturity in years.

        Returns:
            Price : float
                The model-implied bond price.
        """
        B = (1 - np.exp(-kappa_q * tau)) / kappa_q
        A = np.exp((theta_q - sigma ** 2 / (2 * kappa_q ** 2)) * (B - tau) - (sigma ** 2 * B ** 2) / (4 * kappa_q))
        return A * np.exp(-B * r)

    def estimate_risk_premium(self):
        """
        Estimate the risk premium (lambda) by calibrating the risk-neutral parameters
        (kappa_q and theta_q) to the observed 10Y bond yields.

        This version uses only the 10-year yield data. The observed continuously compounded
        yields are converted into discount bond prices via:
            Price = exp(-yield * maturity)

        The objective function minimises the squared error between the observed bond price and
        the Vasicek model-implied price for the 10Y maturity.

        Returns:
            lambda_est : float
                The estimated risk premium, computed as (kappa_q - kappa).
        """
        # Ensure real-world parameters are already estimated
        if self.kappa is None or self.theta is None or self.beta is None:
            raise ValueError("Real-world parameters must be estimated first. Run estimate_params().")

        data = self.data.copy()
        r = data["Short Rate"].values  # short rate time series

        # Use only the 10Y yield data
        maturity = 10
        yield_values = data["10Y"].values
        # Convert continuously compounded yields to bond prices: P = exp(-yield * maturity)
        observed_prices = np.exp(-yield_values * maturity)

        # Define the objective function for the 10Y data
        def objective(params):
            kappa_q, theta_q = params
            error = 0.0
            for i in range(len(r)):
                r_i = r[i]
                P_model = VasicekEstimator.vasicek_bond_price(r_i, kappa_q, theta_q, self.beta, maturity)
                error += (P_model - observed_prices[i]) ** 2
            return error

        # Use an initial guess based on the real-world parameters
        initial_guess = [self.kappa, self.theta]
        bounds = [(1e-4, 5.0), (1e-4, 0.2)]
        res = minimize(objective, initial_guess, bounds=bounds)
        kappa_q, theta_q = res.x

        # The risk premium is given by lambda = kappa_q - kappa
        lambda_est = kappa_q - self.kappa
        print(f"Estimated risk-neutral kappa: {kappa_q:.4f}, theta: {theta_q:.4f}")
        print(f"Estimated risk premium (lambda): {lambda_est:.4f}")
        return lambda_est


import pandas as pd
import numpy as np


class VasicekSimulation:
    def __init__(self, data_file, ppa, kappa, theta, beta):
        """
        Initializes the Vasicek model class.

        Parameters:
            data_file : str
                Path to the interest rate data file.
            ppa : int
                Periods per annum (e.g., 52 for weekly data).
            kappa : float
                Mean-reversion speed.
            theta : float
                Long-term mean.
            beta : float
                Volatility.
        """
        self.ppa = ppa
        self.dt = 1 / ppa
        self.data_file = data_file
        self.rates = self.load_data()
        self.kappa = kappa  # Mean-reversion speed
        self.theta = theta  # Long-term mean
        self.beta = beta  # Volatility

        self.simulation_data = None

    def load_data(self):
        """Load interest rate data from file."""
        # Expects the Excel file to have at least one column, with the first column containing the short rate.
        rates = pd.read_excel(self.data_file, parse_dates=True, index_col=0).iloc[:, 0]
        return rates

    def simulate(self, years: float, num_sim: int, z_mat=None):
        """
        Simulate short-rate paths using the Vasicek model.

        The initial short rate is extracted from the loaded data.

        Parameters:
            years : float
                Simulation horizon in years.
            num_sim : int
                Number of simulation paths.
            z_mat : np.ndarray, optional
                External matrix of standard normal random shocks.
                Should have dimensions (num_sim, num_periods).
                If None, shocks are generated internally.

        Returns:
            np.ndarray
                Simulated short-rate paths.
        """
        # Extract the initial short rate from the data (first entry)
        initial_short_rate = self.rates.iloc[0]

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
                    self.theta +
                    (sim_rates[:, t - 1] - self.theta) * exp_kappa_dt +
                    std_rates * eps[:, t - 1]
            )

        self.simulation_data = sim_rates
        return sim_rates


