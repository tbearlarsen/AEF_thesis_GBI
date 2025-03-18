import os
import subprocess
import pandas as pd

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    #data_folder = os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM")
    output_folder = os.path.join(root, "GBI Optimisation", "Data", "sim_data_yearly")
    data_folder = r"/Users/osito/Library/CloudStorage/OneDrive-SharedLibraries-CBS-CopenhagenBusinessSchool/Thesis - General/Simulation/"

    file_directory = {
        "Ejendomme": "Ejendomme.xlsx",
        "EM Aktier": "Emerging markets aktier.xlsx",
        "EM Statsobl.": "Emerging markets statsobligationer.xlsx",
        "Globale Aktier": "Globale aktier (developed markets).xlsx",
        "Hedgefonde": "Hedgefonde.xlsx",
        "HY Obl.": "High-yield obligationer.xlsx",
        "Infrastruktur": "Infrastruktur.xlsx",
        "IG Obl.": "Investment-grade obligationer.xlsx",
        "Private Equity": "Private equity.xlsx",
        "Stats & RK Obl.": "Stats- og realkreditobligationer.xlsx"
    }

    def yearly_price(file, ppa=52, header=True):
        if header==True:
            yearly_columns = file.iloc[1:, ::ppa]

        else:
            yearly_columns = file.iloc[:, ::ppa]

        return yearly_columns

    total_files = len(file_directory)
    for i, (name, file) in enumerate(file_directory.items(), start=1):
        path = os.path.join(data_folder, file)

        print(f"Converting {name} [{i}/{total_files}]")
        data = pd.read_excel(path, header=None)
        yearly_data = yearly_price(data)
        output_path = os.path.join(output_folder, f"{name}_yearly.xlsx")
        yearly_data.to_excel(output_path, index=False, header=False)
        print(f"Conversion complete\n")

    print("All conversions complete")

if __name__ == "__main__":
    main()