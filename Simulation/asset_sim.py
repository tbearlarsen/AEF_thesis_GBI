import os

from codelib.file_management.dynamic_file_pathing import get_root

root = get_root()
data_folder = os.path.join(root, "GBI Optimisation", "Data")

import sys
sys.path.insert(0,r"C:\Users\thorb\Documents\Github Repositories\AEF_thesis_GBI")

from typing import Union

#numpy for working with matrices, etc.
import numpy as np

#import pandas
import pandas as pd

#plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#scipy for statistics and optimization
from scipy import optimize
from scipy import stats

import cvxpy as cp

#function to perform interpolation

"""
functions from codelib
"""

# functions for calculating moments
from codelib.Johan.statistics import moments as mom

# functions for calculating risk metrics
from codelib.Johan.portfolio_optimization import risk_metrics as rm

# functions for risk budgetting
from codelib.Johan.portfolio_optimization import risk_metrics as rb

# functions for mean-variance optimization
from codelib.Johan.portfolio_optimization import mean_variance as mvo

# cash flows
from codelib.Johan.fixed_income.cash_flows import CashFlow

# predefined plots
from codelib.Johan.visualization.base import fan_chart


"""
Load Data
"""
data_path = os.path.join(data_folder, "Afkvastforventingerne.xlsx")

asset_names = pd.read_excel(data_path, sheet_name='Returns', index_col=0).index.values.flatten()
num_assets = len(asset_names)

asset_indices = {k: i for i, k in enumerate(asset_names)}

mu_linear_1y = pd.read_excel(data_path, sheet_name='Returns', index_col=0).values.flatten()

corr_mat_linear_1y = pd.read_excel(data_path, sheet_name='Correlation', index_col=0).values
corr_mat_linear_1y[np.tril_indices_from(corr_mat_linear_1y, k=-1)] = corr_mat_linear_1y.T[np.tril_indices_from(corr_mat_linear_1y, k=-1)]

vols_linear_1y = pd.read_excel(data_path, sheet_name='Volatility', index_col=0).values.flatten()
costs_linear_1y = pd.read_excel(data_path, sheet_name='Cost', index_col=0).values.flatten()


"""
Calculate covariance matrix
"""

cov_mat_linear_1y = mom.corr_to_cov_matrix(corr_mat_linear_1y, vols_linear_1y)
pd.DataFrame(cov_mat_linear_1y)




