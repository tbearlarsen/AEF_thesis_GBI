import os
import subprocess
import pandas as pd

def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    data_folder=os.path.join(root,"Simulation","Nordnet","simulated_prices","10 year simulation GBM")
    output_folder = os.path.join(root, "Simulation", "Nordnet", "simulated_prices", "10 year simulation GBM", "Yearly Data")

    file_directory = {
        "EUNK": "EUNK_sim_price.xlsx",
        "IUSN": "IUSN_sim_price.xlsx",
        "LYXF": "LYXF_sim_price.xlsx",
        "JGHY": "JGHY_sim_price.xlsx",
        "SXR4": "SXR4_sim_price.xlsx",
        "SXR8": "SXR8_sim_price.xlsx",
        "SYBA": "SYBA_sim_price.xlsx",
        "SYBB": "SYBB_sim_price.xlsx",
        "SYBC": "SYBC_sim_price.xlsx"
    }

    def yearly_price(file,ppa=255):
        yp=file[ppa-1::ppa]
        return yp

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