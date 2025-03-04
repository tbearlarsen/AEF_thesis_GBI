import os
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


class garch_ml_model:
    """
    A class to forecast ETF returns using a hybrid GARCH + ML residual approach
    and Monte Carlo simulation.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing ETF data.
    date_column : str
        The column name representing dates in the CSV.
    price_column : str
        The column name representing ETF prices in the CSV.
    """

    def __init__(self, filepath, date_column, price_column):
        self.filepath = filepath
        self.date_column = date_column
        self.price_column = price_column

        self.data = None
        self.log_returns = None
        self.garch_fit = None
        self.residuals_df = None
        self.X = None  # Features for ML (lagged residuals)
        self.y = None  # Target for ML (residuals)
        self.best_model = None

    def load_data(self):
        """
        Load CSV data, set the date column as the index, and compute log returns.
        """
        self.data = pd.read_csv(self.filepath, parse_dates=[self.date_column], index_col=self.date_column)
        self.data.sort_index(inplace=True)
        self.data['log_returns'] = np.log(self.data[self.price_column] / self.data[self.price_column].shift(1))
        self.log_returns = self.data['log_returns'].dropna()

    def plot_log_returns(self):
        """
        Plot the ETF log returns.
        """
        self.log_returns.plot(title="Log Returns", figsize=(12, 6))
        plt.show()

    def fit_garch_model(self, p=1, q=1):
        """
        Fit a GARCH(p,q) model to the ETF log returns.

        Parameters
        ----------
        p : int, optional
            The number of lagged variance terms (default is 1).
        q : int, optional
            The number of lagged residual terms (default is 1).
        """
        model = arch_model(self.log_returns, vol="GARCH", p=p, q=q, mean="Constant", dist="Normal")
        self.garch_fit = model.fit(disp="off")
        print(self.garch_fit.summary())

    def prepare_ml_dataset(self):
        """
        Extract residuals from the GARCH model and create lagged features for ML.
        """
        residuals = self.garch_fit.resid.dropna()
        df = pd.DataFrame({"residual": residuals})
        df["lag1"] = residuals.shift(1)
        df["lag2"] = residuals.shift(2)
        df = df.dropna()
        self.residuals_df = df
        self.X = df[["lag1", "lag2"]]
        self.y = df["residual"]

    def select_best_ml_model(self):
        """
        Select the best machine learning model for forecasting the GARCH residuals
        using time-series cross-validation. Candidate models: Random Forest,
        Gradient Boosting, and MLP.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        candidate_models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
        }
        model_scores = {}
        for name, model in candidate_models.items():
            scores = cross_val_score(model, self.X, self.y, cv=tscv, scoring="neg_mean_squared_error")
            mse = np.mean(-scores)
            model_scores[name] = mse
            print(f"{name} Mean Squared Error: {mse:.6f}")
        best_model_name = min(model_scores, key=model_scores.get)
        self.best_model = candidate_models[best_model_name]
        self.best_model.fit(self.X, self.y)
        print(f"Selected best model: {best_model_name}")

    def show_model_details(self):
        """
        Print details of the fitted GARCH model and the chosen ML model.
        """
        if self.garch_fit is not None:
            print("GARCH Model Summary:")
            print(self.garch_fit.summary())
        else:
            print("GARCH model has not been fitted yet.")

        if self.best_model is not None:
            print("\nChosen ML Model Details:")
            print(f"Model Type: {type(self.best_model).__name__}")
            print("Parameters:")
            print(self.best_model.get_params())
        else:
            print("ML model has not been selected yet.")

    def run_monte_carlo_simulation(self, horizon=2520, n_simulations=1000):
        """
        Run a Monte Carlo simulation over a specified horizon to generate daily log return paths.

        Parameters
        ----------
        horizon : int, optional
            Forecast horizon in trading days (default is 2520, ~10 years).
        n_simulations : int, optional
            Number of simulation paths to generate (default is 1000).

        Returns
        -------
        simulated_paths : np.array
            Array containing simulated daily log returns.
        """
        sigma_noise = np.std(self.y)
        simulated_paths = np.zeros((n_simulations, horizon))

        for i in range(n_simulations):
            current_lag1 = self.X.iloc[-1]['lag1']
            current_lag2 = self.X.iloc[-1]['lag2']
            path = []
            for t in range(horizon):
                linear_component = self.garch_fit.params['mu']
                features_df = pd.DataFrame([[current_lag1, current_lag2]], columns=self.X.columns)
                ml_residual = self.best_model.predict(features_df)[0]
                noise = np.random.normal(0, sigma_noise)
                daily_return = linear_component + ml_residual + noise
                path.append(daily_return)
                current_lag2 = current_lag1
                current_lag1 = daily_return
            simulated_paths[i, :] = path

        return simulated_paths

    def compute_price_paths(self, simulated_paths):
        """
        Convert simulated daily log returns into price paths.

        Parameters
        ----------
        simulated_paths : np.array
            Array of simulated daily log returns.

        Returns
        -------
        price_paths : np.array
            Array containing the price at each trading day for each simulation path.
        """
        initial_price = self.data[self.price_column].iloc[-1]
        # Compute cumulative log returns along each path
        cumulative_log_returns = np.cumsum(simulated_paths, axis=1)
        # Compute price paths by applying the exponential and scaling by the initial price
        price_paths = initial_price * np.exp(cumulative_log_returns)
        return price_paths

    def plot_simulation_results(self, final_prices):
        """
        Plot the distribution of final ETF prices and print summary statistics.

        Parameters
        ----------
        final_prices : np.array
            Array of final simulated ETF prices.
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(final_prices, bins=50, kde=True)
        plt.xlabel("Final ETF Price")
        plt.ylabel("Frequency")
        plt.title("Distribution of Simulated ETF Prices after 10 Years")
        plt.show()

        print("Mean final ETF price: {:.2f}".format(np.mean(final_prices)))
        print("Median final ETF price: {:.2f}".format(np.median(final_prices)))
        print("5th percentile: {:.2f}".format(np.percentile(final_prices, 5)))
        print("95th percentile: {:.2f}".format(np.percentile(final_prices, 95)))

    def plot_forecast_paths(self, simulated_paths):
        """
        Plot forecast paths (cumulative log returns) over the simulation horizon.

        Parameters
        ----------
        simulated_paths : np.array
            Array containing simulated daily log returns, where each row is one path.
        """
        n_paths, _ = simulated_paths.shape
        cumulative_returns = np.cumsum(simulated_paths, axis=1)
        plt.figure(figsize=(12, 6))
        for i in range(n_paths):
            plt.plot(cumulative_returns[i, :], lw=1, label=f"Path {i + 1}" if n_paths <= 10 else None)
        plt.xlabel("Trading Days")
        plt.ylabel("Cumulative Log Return")
        plt.title("Forecast Paths")
        if n_paths <= 10:
            plt.legend()
        plt.show()

    def export_forecast_to_excel(self, mode, simulated_paths, filename="forecast_results.xlsx", work_dir=None):
        """
        Export the entire price path(s) for each simulation to an Excel file.
        The Excel file is saved in the specified working directory if provided; otherwise,
        it is saved in the current working directory.

        Parameters
        ----------
        mode : str
            "simple" for a one-path simulation,
            "montecarlo" for a Monte Carlo simulation.
        simulated_paths : np.array
            The simulation path(s) in terms of log returns.
        filename : str, optional
            The output Excel file name.
        work_dir : str, optional
            The directory in which to save the Excel file. If not provided, uses os.getcwd().
        """
        # Determine working directory
        if work_dir is None:
            work_dir = os.getcwd()  # Use current working directory if not provided
        else:
            if not os.path.isabs(work_dir):
                work_dir = os.path.join(os.getcwd(), work_dir)
        full_path = os.path.join(work_dir, filename)

        # Compute the price paths from the simulated log returns
        price_paths = self.compute_price_paths(simulated_paths)

        with pd.ExcelWriter(full_path) as writer:
            if mode.lower() == "simple":
                df_prices = pd.DataFrame(price_paths.T, columns=["Price Path"])
                df_prices.to_excel(writer, sheet_name="Price Path", index_label="Trading Day")
            elif mode.lower() == "montecarlo":
                df_prices = pd.DataFrame(price_paths)
                df_prices.to_excel(writer, sheet_name="Price Paths", index=False)
            else:
                raise ValueError("Invalid mode. Choose either 'simple' or 'montecarlo'.")
        print(f"{mode.capitalize()} forecast price paths exported to {full_path}")

    def run_forecast(self, mode="simple", horizon=2520, n_simulations=1000, export=False,
                     filename="forecast_results.xlsx", work_dir=None, show_details=False, plot_paths=False):
        """
        Run a forecast in either "simple" or "montecarlo" mode. This method automatically
        executes all necessary steps if they haven't been run already. In "simple" mode, a single
        simulation path is generated; in "montecarlo" mode, multiple paths are generated.

        Parameters
        ----------
        mode : str
            "simple" for a one-path simulation,
            "montecarlo" for a full Monte Carlo simulation.
        horizon : int, optional
            Forecast horizon in trading days (default is 2520, ~10 years).
        n_simulations : int, optional
            Number of simulation paths (only used for montecarlo mode; default is 1000).
        export : bool, optional
            If True, export the simulation price paths to Excel.
        filename : str, optional
            The Excel filename.
        work_dir : str, optional
            The directory in which to save the Excel file. If not provided, uses the current working directory.
        show_details : bool, optional
            If True, display details of the GARCH and ML models.
        plot_paths : bool, optional
            If True, plot the forecast path(s).

        Returns
        -------
        result : np.array
            In "simple" mode, the simulation path (an array with shape (1, horizon));
            in "montecarlo" mode, an array of simulation paths.
        """
        # Execute necessary steps if not already done
        if self.data is None:
            self.load_data()
        if self.log_returns is None:
            self.log_returns = self.data['log_returns'].dropna()
        if self.garch_fit is None:
            self.fit_garch_model()
        if self.X is None or self.y is None:
            self.prepare_ml_dataset()
        if self.best_model is None:
            self.select_best_ml_model()

        if show_details:
            self.show_model_details()

        if mode.lower() == "simple":
            sim_paths = self.run_monte_carlo_simulation(horizon=horizon, n_simulations=1)
            if plot_paths:
                self.plot_forecast_paths(sim_paths)
            if export:
                self.export_forecast_to_excel("simple", sim_paths, filename=filename, work_dir=work_dir)
            return sim_paths
        elif mode.lower() == "montecarlo":
            sim_paths = self.run_monte_carlo_simulation(horizon=horizon, n_simulations=n_simulations)
            if plot_paths:
                if n_simulations > 10:
                    idx = np.random.choice(n_simulations, 10, replace=False)
                    self.plot_forecast_paths(sim_paths[idx])
                else:
                    self.plot_forecast_paths(sim_paths)
            if export:
                self.export_forecast_to_excel("montecarlo", sim_paths, filename=filename, work_dir=work_dir)
            return sim_paths
        else:
            raise ValueError("Invalid mode. Choose either 'simple' or 'montecarlo'.")

