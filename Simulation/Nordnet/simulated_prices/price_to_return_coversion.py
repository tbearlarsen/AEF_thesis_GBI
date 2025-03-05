import os
import subprocess
import pandas as pd

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder=os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM", "Yearly Data")
    output_folder = os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM", "Yearly Data")

    file_directory = {
        "EUNK": "EUNK_yearly.xlsx",
        "IUSN": "IUSN_yearly.xlsx",
        "LYXF": "LYXF_yearly.xlsx",
        "JGHY": "JGHY_yearly.xlsx",
        "SXR4": "SXR4_yearly.xlsx",
        "SXR8": "SXR8_yearly.xlsx",
        "SYBA": "SYBA_yearly.xlsx",
        "SYBB": "SYBB_yearly.xlsx",
        "SYBC": "SYBC_yearly.xlsx"
    }

    total_files=len(file_directory)
    for i, (name, file_name) in enumerate(file_directory.items(), start=1):
        file_path = os.path.join(data_folder, file_name)

        print(f"Computing yearly returns for [{name}] [{i}/{total_files}]")
        data=pd.read_excel(file_path,header=None)
        returns=data.pct_change().dropna()
        output_path=os.path.join(output_folder, f"{name}_yearly_returns.xlsx")
        returns.to_excel(output_path, index=False, header=False)
        print(f"Conversion complete\n"
              f"Exported to {output_path}\n\n")

    print("All conversions complete")

if __name__ == "__main__":
    main()