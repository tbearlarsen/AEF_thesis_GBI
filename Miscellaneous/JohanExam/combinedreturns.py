import pandas as pd

# List of files (without .csv extension)
files = [
    "EUNK_iShares_Core_MSCI_Europe_UCITS_ETF_EUR_Acc",
    "IUSN_iShares_MSCI_World_Small_Cap_UCITS_ETF_USD_Acc",
    "JGHY_JPM_Global_High_Yield_Corporate_Bond_Multi-Factor_UCITS_ETF_USD_acc",
    "SXR4_iShares_MSCI_USA_UCITS_ETF_USD_Acc",
    "SYBB_SPDR_Bloomberg_Euro_Government_Bond_UCITS_ETF_Dist",
    "SYBC_SPDR_Bloomberg_Euro_Corporate_Bond_UCITS_ETF_Dist"
]

data_dict = {}
for f in files:
    csv_path = f"../Simulation/Data/{f}.csv"  # Adjust path if needed
    df = pd.read_csv(csv_path)
    df.rename(columns={"Dato": "Date", "Slutkurs": "ClosedPrice"}, inplace=True)
    data_dict[f] = df

combined_df = None
for name, df in data_dict.items():
    short_name = name.split("_")[0]  # e.g. "EUNK", "IUSN", etc.
    temp_df = df[["Date", "ClosedPrice"]].copy()
    temp_df.rename(columns={"ClosedPrice": f"{short_name}_ClosedPrice"}, inplace=True)

    # Use how="inner" so only dates present in *all* CSVs remain
    if combined_df is None:
        combined_df = temp_df
    else:
        combined_df = pd.merge(combined_df, temp_df, on="Date", how="inner")

# Export combined table to a CSV file
combined_df.to_csv("combined_table.csv", index=False)

print("CSV file saved as combined_table.csv")
print(combined_df.head())

