import pandas as pd
import os
from codelib.file_management.dynamic_file_pathing import get_root
import numpy as np

root = get_root()
data_path = os.path.join(root, "Simulation", "Data", "JGHY_JPM_Global_High_Yield_Corporate_Bond_Multi-Factor_UCITS_ETF_USD_acc.csv")

data = pd.read_csv(data_path, parse_dates=["Dato"], index_col="Dato")
prices = data["Slutkurs"]
returns = prices/ prices.shift(1) - 1
returns = returns.dropna()