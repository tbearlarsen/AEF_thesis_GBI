import pandas as pd
from Models.geometric_brownian_motion_model import GBMSimulator

def main():
    file_paths = {
        "EUNK": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/EUNK_iShares_Core_MSCI_Europe_UCITS_ETF_EUR_Acc.csv",
        "SXR8": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/SXR8_iShares_Core_S&P_500_ETF_USD_Acc_EUR.csv",
        "IUSN": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/IUSN_iShares_MSCI_World_Small_Cap_UCITS_ETF_USD_Acc.csv",
        "JGHY": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/JGHY_JPM_Global_High_Yield_Corporate_Bond_Multi-Factor_UCITS_ETF_USD_acc.csv",
        "LYXF": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/LYXF_Amundi_Euro_Government_Bond_15plusY_UCITS_ETF_Acc.csv",
        "SXR4": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/SXR4_iShares_MSCI_USA_UCITS_ETF_USD_Acc.csv",
        "SYBA": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/SYBA_SPDR_Bloomberg_Euro_Aggregate_Bond_UCITS_ETF_Dist.csv",
        "SYBB": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/SYBB_SPDR_Bloomberg_Euro_Government_Bond_UCITS_ETF_Dist.csv",
        "SYBC": "/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/SYBC_SPDR_Bloomberg_Euro_Corporate_Bond_UCITS_ETF_Dist.csv"
    }

    prices = pd.DataFrame()

    for name, path in file_paths.items():
        data=pd.read_csv(path, index_col=0, parse_dates=True)
        data=data[data.index.year < 2025]
        prices[name]=data["Slutkurs"]

    for column in prices.columns:
        print(f"Starting simulation for {column}\n")
        column_data=prices[column].dropna()
        GBM = GBMSimulator(column_data, 255)
        simulated_prices = GBM.simulate_gbm(10, 1000)
        simulated_prices=pd.DataFrame(simulated_prices)
        simulated_prices.to_excel(f"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Nordnet/simulated_prices/{column}_sim_price.xlsx", sheet_name=column,index=False,header=False)
        print(f"Simulation for {column} completed\n")

if __name__ == "__main__":
    main()