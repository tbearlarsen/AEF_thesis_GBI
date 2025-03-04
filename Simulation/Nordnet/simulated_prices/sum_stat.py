import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_paths = {
        "EUNK": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/EUNK_yearly_conversion.xlsx",
        "IUSN": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/IUSN_yearly_conversion.xlsx",
        "LYXF": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/LYXF_yearly_conversion.xlsx",
        "SXR4": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SXR4_yearly_conversion.xlsx",
        "SXR8": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SXR8_yearly_conversion.xlsx",
        "SYBA": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBA_yearly_conversion.xlsx",
        "SYBB": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBB_yearly_conversion.xlsx",
        "SYBC": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBC_yearly_conversion.xlsx"
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

    for name, path in file_paths.items():
        data=pd.read_excel(path, header=None)

        print(f"Processing {name}")
        row_summaries=[]
        for index, row in data.iterrows():
            stats = sum_stat(row)
            row_summaries.append(stats)
            row_sum = pd.DataFrame(row_summaries)
        row_sum.to_excel(f"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/{name}_sum_stat.xlsx", index=False)
        print(f"Finished processing {name}\n")

if __name__ == "__main__":
    main()