from codelib.file_management.dynamic_file_pathing import get_root
from codelib.Models.ZCB import ZCBYieldTimeMaturity
import os
import pandas as pd

def main():
    root = get_root()
    short_rate_path = os.path.join(root, "Miscellaneous", "Archive", "Data", "short_rate.xlsx")
    risk_premium = -0.2

    simulator = ZCBYieldTimeMaturity(short_rate_path,
                                     risk_premium,
                        252,
                                     222,
                                     10,
                                        range(1, 11),
                                        range(1, 21)
                                     )
    yields_dict = simulator.simulate_yields()
    year_5 = simulator.get_yields_for_year(10)

    return yields_dict

if __name__ == '__main__':
    simulated_yields = main()

    year_5 = pd.DataFrame(simulated_yields[5])
    print(type(simulated_yields))
    print(len(simulated_yields))
    print(simulated_yields)

