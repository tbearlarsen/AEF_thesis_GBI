import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

# ------------------------------
# Data Loading & Log Returns
# ------------------------------
data = pd.read_csv(
    r"C:\Users\thorb\Documents\Github Repositories\AEF_msc_thesis_GBI\Simulation\Data\iShares_Core_S&P_500_ETF_USD_Acc_EUR.csv",
    parse_dates=['Dato'], index_col='Dato')
data['log_returns'] = np.log(data["Slutkurs"] / data["Slutkurs"].shift(1))
returns = data["log_returns"].dropna()

# Plot log returns
returns.plot(title="Log returns", figsize=(12, 6))
plt.show()

# ------------------------------
# GARCH(1,1) Model Fitting
# ------------------------------
gm = arch_model(returns, vol="GARCH", p=1, q=1, mean="Constant", dist="Normal")
g_fit = gm.fit(disp="off")
print(g_fit.summary())

# ------------------------------
# Residual Extraction & Feature Engineering for ML
# ------------------------------
residuals = g_fit.resid.dropna()
residuals_df = pd.DataFrame({"residuals": residuals})
residuals_df["lag1"] = residuals.shift(1)
residuals_df["lag2"] = residuals.shift(2)
residuals_df = residuals_df.dropna()

X = residuals_df[["lag1", "lag2"]]
y = residuals_df["residuals"]

# ------------------------------
# Machine Learning Model Selection
# ------------------------------
tscv = TimeSeriesSplit(n_splits=5)
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
}

model_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
    model_scores[name] = np.mean(-scores)
    print(f"{name} Mean Squared Estimator: {model_scores[name]:.6f}")

best_model_name = min(model_scores, key=model_scores.get)
best_model = models[best_model_name]
best_model.fit(X, y)
print(f"Best model: {best_model_name}")

# ------------------------------
# Forecast the residual for the next time period
# ------------------------------
# Instead of converting to a numpy array, we use a one-row DataFrame to retain feature names.
last_values_df = X.iloc[-1:].copy()
residual_forecast = best_model.predict(last_values_df)[0]
print(f"Forecasted residual for the next period: {residual_forecast:.6f}")

# ------------------------------
# One-step ahead Hybrid Forecast (GARCH + ML)
# ------------------------------
g_forecast = g_fit.forecast(horizon=1)
linear_forecast = g_forecast.mean.iloc[-1, 0]
hybrid_forecast = linear_forecast + residual_forecast
print(f"Hybrid forecast (GARCH + ML residual): {hybrid_forecast:.6f}")

# ------------------------------
# Monte Carlo Simulation Parameters
# ------------------------------
horizon = 2520  # 10 years ~ 2520 trading days
n_simulations = 1000
sigma_noise = np.std(y)

# Initialize array to store simulation paths (daily log returns)
simulated_paths = np.zeros((n_simulations, horizon))

# ------------------------------
# Monte Carlo Simulation Loop
# ------------------------------
for i in range(n_simulations):
    # Initialize lag features with the last available values
    current_lag1 = X.iloc[-1]['lag1']
    current_lag2 = X.iloc[-1]['lag2']
    simulated_returns = []

    for t in range(horizon):
        # GARCH (Linear) Component using the constant mean (mu)
        linear_forecast_t = g_fit.params['mu']

        # ML Residual Forecast: prepare features as a DataFrame to preserve feature names
        features_df = pd.DataFrame([[current_lag1, current_lag2]], columns=X.columns)
        ml_forecast = best_model.predict(features_df)[0]

        # Add random noise (assumed normally distributed)
        noise = np.random.normal(0, sigma_noise)

        # Combine components to form the forecast return for this day
        forecast_return = linear_forecast_t + ml_forecast + noise
        simulated_returns.append(forecast_return)

        # Update lag features for next iteration
        current_lag2 = current_lag1
        current_lag1 = forecast_return

    simulated_paths[i, :] = simulated_returns

# ------------------------------
# Convert Simulated Log Returns to Final ETF Prices
# ------------------------------
S0 = data["Slutkurs"].iloc[-1]
final_values = S0 * np.exp(np.sum(simulated_paths, axis=1))

# ------------------------------
# Visualize & Summarize the Simulation Results
# ------------------------------
plt.figure(figsize=(12, 6))
sns.histplot(final_values, bins=50, kde=True)
plt.xlabel("Final ETF Price")
plt.ylabel("Frequency")
plt.title("Distribution of Simulated ETF Prices after 10 Years")
plt.show()

print("Mean final ETF price: {:.2f}".format(np.mean(final_values)))
print("Median final ETF price: {:.2f}".format(np.median(final_values)))
print("5th percentile: {:.2f}".format(np.percentile(final_values, 5)))
print("95th percentile: {:.2f}".format(np.percentile(final_values, 95)))
