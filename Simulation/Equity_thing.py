import pandas as pd
import os
from codelib.file_management.dynamic_file_pathing import get_root
import numpy as np

root = get_root()
data_path = os.path.join(root, "Simulation", "Data", "JGHY_JPM_Global_High_Yield_Corporate_Bond_Multi-Factor_UCITS_ETF_USD_acc.csv")

data = pd.read_csv(data_path, parse_dates=["Dato"], index_col="Dato")
data.index = pd.to_datetime(data.index, format="%d/%m/%Y")

prices = data["Slutkurs"]
daily_returns = prices.pct_change().dropna()
daily_returns.mean()
daily_returns.std()

#######
short_rate = pd.read_excel(os.path.join(root, "Miscellaneous", "Archive", "Data", "short_rate.xlsx"), parse_dates=["Date"], index_col="Date")

# Align the indices of prices and short_rate
aligned_data = pd.concat([prices, short_rate], axis=1, join='inner')

# Drop any rows with missing values
aligned_data.dropna(inplace=True)

correlation = aligned_data.corr().iloc[0, 1]

frequency = pd.infer_freq(aligned_data.index)