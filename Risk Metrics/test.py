import pandas as pd
import numpy as np
import os
from codelib.file_management.dynamic_file_pathing import get_root
#import ace_tools as tools

root = get_root()
folder = os.path.join(root, "Risk Metrics")

return_files = {
    "EM Equities":        "EM Aktier_yearly_returns.xlsx",
    "EM Sovereign":       "EM Statsobl._yearly_returns.xlsx",
    "Global Equities":    "Globale Aktier_yearly_returns.xlsx",
    "High Yield":         "HY Obl._yearly_returns.xlsx",
    "Investment Grade":   "IG Obl._yearly_returns.xlsx",
    "Gov & Mortgage":     "Stats & RK Obl._yearly_returns.xlsx",
}

ret_dfs = {asset: pd.read_excel(f"{folder}/{fname}")
           for asset, fname in return_files.items()}
assets = list(return_files.keys())
N_full, T = ret_dfs[assets[0]].shape


# 2. Load and clean path-specific weights
weights_long = pd.read_csv(f"{folder}/P1_port_w.csv")
# Map 'Asset' names to our return asset keys
long_to_key = {
    'Emerging Markets - Equities':         'EM Equities',
    'Developed Markets - Equities':        'Global Equities',
    'Emerging Markets State - Obligations':'EM Sovereign',
    'High Yield Bonds - Obligations':      'High Yield',
    'Investment Grade Bonds - Obligations':'Investment Grade',
    'Government ZC Bonds - Obligations':   'Gov & Mortgage'
}
weights_long['Asset_Key'] = weights_long['Asset'].map(long_to_key)
# Filter out initial year if extra
years_all = sorted(weights_long['Year'].unique())
if len(years_all) == T + 1:
    weights_long = weights_long[weights_long['Year'] != years_all[0]]
# Confirm periods count
years = sorted(weights_long['Year'].unique())
assert len(years) == T, "Year count mismatch"

# Identify path IDs
path_ids = sorted(weights_long['Path_ID'].unique())
N = len(path_ids)

# 3. Build return array (N, T, 6) then (T, N, 6)
ret_vals = np.stack([ret_dfs[a].values[:N, :] for a in assets], axis=2)
R = ret_vals.transpose(1, 0, 2)

# 4. Pivot weights_long to (N, T, 6)
w_pivot = (
    weights_long
    .sort_values(['Path_ID', 'Year'])
    .pivot(index=['Path_ID', 'Year'], columns='Asset_Key', values='Weight')
)
w_pivot = w_pivot[assets]  # reorder columns to match 'assets' list
w_arr = w_pivot.values.reshape(N, T, len(assets))
weights = w_arr.transpose(1, 0, 2)

# 5. Compute portfolio returns and wealth
assert np.allclose(weights.sum(axis=2), 1, atol=1e-8)
r_port = (R * weights).sum(axis=2)
wealth = np.empty((T + 1, N))
wealth[0] = 1.0
wealth[1:] = 1.0 * np.cumprod(1 + r_port, axis=0)
terminal = wealth[-1]

# 6. Compute risk metrics
mean_W = terminal.mean()
std_W = terminal.std(ddof=1)
alpha = 0.05
sorted_W = np.sort(terminal)
j = int(np.ceil(alpha * N)) - 1
VaR = mean_W - sorted_W[j]
CVaR = mean_W - sorted_W[:j+1].mean()
n = N
m3 = ((terminal - mean_W) ** 3).sum() / n
m4 = ((terminal - mean_W) ** 4).sum() / n
skew = (n/((n-1)*(n-2))) * m3 / std_W**3
kurt = ((n*(n+1))/((n-1)*(n-2)*(n-3))) * m4 / std_W**4 \
       - (3*(n-1)**2)/((n-2)*(n-3))

# 7. Display results
metrics_df = pd.DataFrame({
    "Mean":      [mean_W],
    "Std Dev":   [std_W],
    "VaR 5%":    [VaR],
    "CVaR 5%":   [CVaR],
    "Skewness":  [skew],
    "Kurtosis":  [kurt]
}, index=["Portfolio"])

#tools.display_dataframe_to_user("Classical Distribution-Based Risk Metrics", metrics_df)

print(metrics_df.to_markdown())