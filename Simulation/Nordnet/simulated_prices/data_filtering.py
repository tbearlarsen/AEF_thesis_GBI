import pandas as pd

def main():
    file_paths = {
        "EUNK": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/EUNK_sim_price.xlsx",
        "IUSN": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/IUSN_sim_price.xlsx",
        "LYXF": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/LYXF_sim_price.xlsx",
        "SXR4": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SXR4_sim_price.xlsx",
        "SXR8": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SXR8_sim_price.xlsx",
        "SYBA": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBA_sim_price.xlsx",
        "SYBB": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBB_sim_price.xlsx",
        "SYBC": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/SYBC_sim_price.xlsx"
    }

    ppa=255

    def yearly_price(file,ppa):
        yp=file[ppa-1::ppa]
        return yp

    for name, path in file_paths.items():
        print(f"Cleaning {name}")
        data=pd.read_excel(path,header=None)
        yearly_data=yearly_price(data,ppa)
        yearly_data.to_excel(f"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_thesis_GBI/Simulation/Nordnet/simulated_prices/10 year simulation GBM/{name}_yearly_conversion.xlsx",index=False,header=False)
        print(f"Cleaning complete\n")

if __name__ == "__main__":
    main()