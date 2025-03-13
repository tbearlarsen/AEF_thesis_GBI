import os
import subprocess
import pandas as pd
from codelib.Models.gbi_optimiser import PortfolioOptimizer
from codelib.Models.gbi_optimiser import PortfolioData

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
data_folder = os.path.join(root, "GBI Optimisation", "data")
output_folder = os.path.join(root, "GBI Optimisation")

file_directory = {
        "CapMktExp": "Capital Market Expectations.csv",
        "Corr": "Correlations - Kitchen Sink.csv",
        "Goals": "Example Goal Details.csv"
    }

# Example usage – adjust file paths as needed.
Goals=os.path.join(data_folder, file_directory["Goals"])
CapMktExp=os.path.join(data_folder, file_directory["CapMktExp"])
Corr=os.path.join(data_folder, file_directory["Corr"])
pool=10000

data = PortfolioData(Goals, CapMktExp, Corr)
optimiser = PortfolioOptimizer(data, pool)

opt_w, opt_agg_p = optimiser.run()
