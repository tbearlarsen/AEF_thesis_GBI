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
pool=4654000

optimiser = PortfolioOptimizer(Goals, CapMktExp, Corr, pool)
optimiser.run()

optimiser.optimize_within_goal_allocation()
optimiser.simulate_across_goal_allocation()
optimiser.compute_aggregate_portfolio()
# Plot all goals; you can change the argument to a specific goal or list of goals.
optimiser.plot_goal_allocation("all")
optimiser.print_results()
# Export all outputs.
optimiser.export_all_outputs()