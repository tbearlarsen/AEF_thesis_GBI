import numpy as np
import pandas as pd

rates = pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/short_rate.xlsx", parse_dates=[0], index_col=0)

years = rates.index.year
total_days=len(rates)
unique_years=len(np.unique(years))
average_days_per_year=total_days/unique_years

