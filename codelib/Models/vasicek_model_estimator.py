import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


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

# Example usage:
# data_file = "path/to/your/excel_file.xlsx"
# model = VasicekModel(data_file, ppa=12)  # ppa=12 for monthly data
# model.estimate_params()
# risk_premium = model.estimate_risk_premium()
