import os
import pandas as pd
from codelib.file_management.dynamic_file_pathing import get_root


def main():
    root = get_root()
    data_folder = os.path.join(root, "GBI Optimisation", "Data", "sim_data_yearly")
    output_folder = os.path.join(root, "GBI Optimisation", "Data", "sim_data_yearly")

    file_directory = {
        "Ejendomme": "Ejendomme_yearly_returns.xlsx",
        "EM Aktier": "EM Aktier_yearly_returns.xlsx",
        "EM Statsobl.": "EM Statsobl._yearly_returns.xlsx",
        "Globale Aktier": "Globale Aktier_yearly_returns.xlsx",
        "Hedgefonde": "Hedgefonde_yearly_returns.xlsx",
        "HY Obl.": "HY Obl._yearly_returns.xlsx",
        "IG Obl.": "IG Obl._yearly_returns.xlsx",
        "Infrastruktur": "Infrastruktur_yearly_returns.xlsx",
        "Private Equity": "Private Equity_yearly_returns.xlsx",
        "Stats & RK Obl.": "Stats & RK Obl._yearly_returns.xlsx"
    }

    def sum_stat(series):
        return {
            'Column': series.name,
            'Mean': series.mean(),
            'Median': series.median(),
            'Std Dev': series.std(),
            'Variance': series.var(),
            'Min': series.min(),
            '25th Percentile': series.quantile(0.25),
            '50th Percentile': series.quantile(0.50),
            '75th Percentile': series.quantile(0.75),
            'Max': series.max()
        }

    total_files = len(file_directory)
    for i, (name, file) in enumerate(file_directory.items(), start=1):
        path = os.path.join(data_folder, file)
        data = pd.read_excel(path, header=None)
        print(f"Processing [{name}] [{i}/{total_files}]")

        col_summaries = []
        for col in data.columns:
            stats = sum_stat(data[col])
            col_summaries.append(stats)
        col_sum = pd.DataFrame(col_summaries)

        output_path = os.path.join(output_folder, f"{name}_sum_stat.xlsx")
        col_sum.to_excel(output_path, index=False)
        print(f"Finished processing {name}\n")

    print("Summary statistics for all files have been computed")


if __name__ == "__main__":
    main()