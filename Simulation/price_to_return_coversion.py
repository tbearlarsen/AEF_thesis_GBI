import os
import subprocess
import pandas as pd

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder=os.path.join(root, "GBI Optimisation", "Data", "sim_data_yearly")
    output_folder=os.path.join(root, "GBI Optimisation", "Data", "sim_data_yearly")

    file_directory = {
        "Ejendomme": "Ejendomme_yearly.xlsx",
        "EM Aktier": "EM Aktier_yearly.xlsx",
        "EM Statsobl.": "EM Statsobl._yearly.xlsx",
        "Globale Aktier": "Globale Aktier_yearly.xlsx",
        "Hedgefonde": "Hedgefonde_yearly.xlsx",
        "HY Obl.": "HY Obl._yearly.xlsx",
        "IG Obl.": "IG Obl._yearly.xlsx",
        "Infrastruktur": "Infrastruktur_yearly.xlsx",
        "Private Equity": "Private Equity_yearly.xlsx",
        "Stats & RK Obl.": "Stats & RK Obl._yearly.xlsx"
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

    print("ALL CONVERSIONS COMPLETE")

if __name__ == "__main__":
    main()