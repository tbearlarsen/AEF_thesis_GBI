import os
import subprocess
import pandas as pd

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder=os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM", "Yearly Data")
    output_folder = os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM", "Yearly Data")

    file_directory = {
        "EUNK": "EUNK_yearly_returns.xlsx",
        "IUSN": "IUSN_yearly_returns.xlsx",
        "LYXF": "LYXF_yearly_returns.xlsx",
        "JGHY": "JGHY_yearly_returns.xlsx",
        "SXR4": "SXR4_yearly_returns.xlsx",
        "SXR8": "SXR8_yearly_returns.xlsx",
        "SYBA": "SYBA_yearly_returns.xlsx",
        "SYBB": "SYBB_yearly_returns.xlsx",
        "SYBC": "SYBC_yearly_returns.xlsx"
    }

    def sum_stat(row):
        return {
            'Mean': row.mean(),
            'Median': row.median(),
            'Std Dev': row.std(),
            'Variance': row.var(),
            'Min': row.min(),
            '25th Percentile': row.quantile(0.25),
            '50th Percentile': row.quantile(0.50),
            '75th Percentile': row.quantile(0.75),
            'Max': row.max()
        }

    total_files=len(file_directory)
    for i, (name, file) in enumerate(file_directory.items(), start=1):
        path = os.path.join(data_folder, file)
        data=pd.read_excel(path, header=None)
        print(f"Processing [{name}] [{i}/{total_files}]")
        row_summaries=[]
        for index, row in data.iterrows():
            stats = sum_stat(row)
            row_summaries.append(stats)
            row_sum = pd.DataFrame(row_summaries)
        output_path=os.path.join(output_folder, f"{name}_sum_stat.xlsx")
        row_sum.to_excel(output_path, index=False)
        print(f"Finished processing {name}\n")

    print("Summary statistics for all files have been computed")

if __name__ == "__main__":
    main()