import numpy as np
import pandas as pd
import yfinance as yf
import datetime

# -----------------------------
# 1. Download historical data
# -----------------------------
# Ticker list: '^GSPC' for the S&P500, '^TNX' for the CBOE 10-Year Treasury Note Yield, and 'IBIT' as a placeholder.
tickers = ['^GSPC', '^TNX', 'IBIT']

# Define historical period (adjust as needed)
start_date = '2010-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Download adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Drop rows with missing values
data.dropna(inplace=True)

# -----------------------------
# 2. Compute historical correlation
# -----------------------------
# Compute daily returns
returns = data.pct_change().dropna()
# Calculate the correlation matrix
corr_matrix = returns.corr()
print("Historical Correlation Matrix:")
print(corr_matrix)

# ---------------------------------------------
# 3. Simulate future correlation matrices (2025-2060)
# ---------------------------------------------
# We'll simulate the evolution of the three off-diagonal correlations.
# First, extract current correlations:
r_SP_Treasury = corr_matrix.loc['^GSPC', '^TNX']
r_SP_IBIT     = corr_matrix.loc['^GSPC', 'IBIT']
r_Treasury_IBIT = corr_matrix.loc['^TNX', 'IBIT']

# Define functions for Fisher z-transformation and its inverse.
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# Transform current correlations:
z_SP_Treasury = fisher_z(r_SP_Treasury)
z_SP_IBIT     = fisher_z(r_SP_IBIT)
z_Treasury_IBIT = fisher_z(r_Treasury_IBIT)

# Simulation parameters for the AR(1) process:
rho = 0.9      # persistence
sigma = 0.05   # volatility of the innovation
np.random.seed(42)  # For reproducibility

years = np.arange(2025, 2061)
simulated_corrs = {}

# Function to check if a matrix is positive definite.
def isPD(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# Function to project a symmetric matrix to the nearest positive definite matrix.
def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

# Initialize current z-values with the historical values:
current_z_SP_Treasury = z_SP_Treasury
current_z_SP_IBIT     = z_SP_IBIT
current_z_Treasury_IBIT = z_Treasury_IBIT

# Simulate for each future year
for year in years:
    # Update each z with an AR(1) step:
    current_z_SP_Treasury = rho * current_z_SP_Treasury + np.random.normal(0, sigma)
    current_z_SP_IBIT     = rho * current_z_SP_IBIT + np.random.normal(0, sigma)
    current_z_Treasury_IBIT = rho * current_z_Treasury_IBIT + np.random.normal(0, sigma)
    
    # Convert back to correlation coefficients:
    sim_r_SP_Treasury = inverse_fisher_z(current_z_SP_Treasury)
    sim_r_SP_IBIT     = inverse_fisher_z(current_z_SP_IBIT)
    sim_r_Treasury_IBIT = inverse_fisher_z(current_z_Treasury_IBIT)
    
    # Build the 3x3 correlation matrix:
    sim_corr = np.array([
        [1.0, sim_r_SP_Treasury, sim_r_SP_IBIT],
        [sim_r_SP_Treasury, 1.0, sim_r_Treasury_IBIT],
        [sim_r_SP_IBIT, sim_r_Treasury_IBIT, 1.0]
    ])
    
    # Ensure the simulated matrix is positive definite.
    if not isPD(sim_corr):
        sim_corr = nearestPD(sim_corr)
    
    simulated_corrs[year] = sim_corr

# Display simulated correlation matrices
print("\nSimulated Future Correlation Matrices (2025-2060):")
for year, matrix in simulated_corrs.items():
    print(f"\nYear: {year}")
    print(matrix)


