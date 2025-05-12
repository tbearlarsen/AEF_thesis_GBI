import pandas as pd
import os
from codelib.file_management.dynamic_file_pathing import get_root

root = get_root()

# 1) Load & aggregate your base scenario
base_path = os.path.join(root, "Risk Metrics", "Stress_testing", "base.csv")
base_df   = pd.read_csv(base_path, header=0, index_col=0)
# take mean across any replicate rows for each goal, then divide by total (50)
base_SR = base_df.groupby(level=0)['Successes'].mean() / 50

# 2) Build your nested dict of aggregated success‐rates
sources = {
    'Contr':  os.path.join(root, "Risk Metrics", "Stress_testing", "Contribution"),
    'ExpRet': os.path.join(root, "Risk Metrics", "Stress_testing", "Expected Return"),
    'Wealth': os.path.join(root, "Risk Metrics", "Stress_testing", "Wealth"),
}

scenarios = {
    "-10%": "10m.csv", "-25%": "25m.csv", "-50%": "50m.csv",
    "+10%": "10p.csv", "+25%": "25p.csv", "+50%": "50p.csv"
}

success_rates = {}
for src_name, folder in sources.items():
    success_rates[src_name] = {}
    for scen_name, fname in scenarios.items():
        df   = pd.read_csv(os.path.join(folder, fname), header=0, index_col=0)
        # aggregate replicates per goal
        sr   = df.groupby(level=0)['Successes'].mean() / 50
        success_rates[src_name][scen_name] = sr

# 1) Base success rates (as %)
print("=== Base Success Rates (%) ===")
print((base_SR).round(2))

# 2) Success rates in each stress‐scenario, per source
for src, scen_dict in success_rates.items():
    df_rates = pd.DataFrame(scen_dict)     # convert fraction → %
    df_rates = df_rates.round(2)                 # e.g. 34.56
    print(f"\n=== {src} Success Rates (%) ===")
    print(df_rates)

# 3) First‐order stress‐test: one difference per goal
stress_diff = {}
for src, scen_dict in success_rates.items():
    stress_diff[src] = {}
    for scen, sr in scen_dict.items():
        # now sr and base_SR both have unique index A,B,C,D
        stress_diff[src][scen] = (sr - base_SR)   # in %-points

for src, diffs in stress_diff.items():
    df = pd.DataFrame(diffs).round(4)
    print(f"\n=== {src} (pp change vs base) ===")
    print(df)


# 4) Stress_testing: average of the two stress‐scenarios
delta_pairs = [("-10%", "+10%"), ("-25%", "+25%"), ("-50%", "+50%")]

all_fragility = {}
for src, scen_dict in success_rates.items():
    all_fragility[src] = {}
    for neg, pos in delta_pairs:
        H = (scen_dict[neg] + scen_dict[pos]) / 2 - base_SR
        all_fragility[src][f"{neg}/{pos}"] = H

fragility_full_df = pd.DataFrame({
    f"{src} ({pair})": series
    for src, pairs in all_fragility.items()
    for pair, series in pairs.items()
}).round(4)

print(fragility_full_df)


fragility_pct = fragility_full_df * 100

# pandas ≥1.0 has to_markdown
print(fragility_pct.to_markdown(
    tablefmt="github",
    headers="keys",
    floatfmt=".2f"
))

import matplotlib.pyplot as plt

# Assume `fragility_full_df` is already defined: index = goals, columns like 'Contr (-10%/+10%)', etc.

# Parse variable names and Δ magnitudes from column labels
column_info = [(col.split()[0], col[col.find('(')+1:col.find(')')])
               for col in fragility_full_df.columns]
# Map each variable to its list of (delta, column index)
var_to_entries = {}
for idx, (var, pair) in enumerate(column_info):
    delta = abs(float(pair.split('/')[0].strip('%+')))
    var_to_entries.setdefault(var, []).append((delta, idx))

# Generate a separate plot for each goal
for goal in fragility_full_df.index:
    plt.figure()
    for var, entries in var_to_entries.items():
        # Sort entries by delta magnitude
        entries_sorted = sorted(entries, key=lambda x: x[0])
        xs = [0] + [d for d, _ in entries_sorted]  # include Δ = 0
        ys = [0] + [fragility_full_df.iloc[fragility_full_df.index.get_loc(goal), idx]
                   for _, idx in entries_sorted]
        plt.plot(xs, ys, marker='o', label=var)
    plt.axhline(0)
    plt.xlabel('Relative change in input (Δ %)')  # no explicit colours
    plt.ylabel('Stress_testing measure $H$')
    plt.title(f'Stress_testing profile for goal {goal}')
    plt.legend()
    plt.show()
