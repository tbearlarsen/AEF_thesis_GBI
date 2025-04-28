#%%
import pandas as pd
import os
import subprocess
#%%
# Get repo root and set folders
root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
data_folder = os.path.join(root, "GBI Optimisation", "data")
output_folder = os.path.join(root, "GBI Optimisation")

# Get excel file and sheets
excel_path = os.path.join(data_folder, "salaryP1 Nominal.xlsx")

# Load your salary data
salary_data = pd.read_excel(excel_path)  # Adjust path if needed

# Rename columns for clarity
salary_data.columns = ['Year', 'Gross_Salary_DKK']
#%%
# Constants
inflation_rate = 0.02
nominal_investment_return = 0.084
real_investment_return = nominal_investment_return - inflation_rate
tax_on_gains = 0.153
ratepension_contrib_own = 0.05
ratepension_contrib_company = 0.10
ratepension_total_contrib = ratepension_contrib_own + ratepension_contrib_company
initial_ratepension_yearly_cost = 888  # DKK
aldersopsparing_contrib = 9400  # DKK per year
avg_post_tax_rate = 0.63  # average net salary percentage after tax

#%%

# Initialize columns
salary_data['Ratepension_Contribution'] = salary_data['Gross_Salary_DKK'] * ratepension_total_contrib
salary_data['Aldersopsparing_Contribution'] = aldersopsparing_contrib
salary_data['Ratepension_Balance'] = 0.0
salary_data['Aldersopsparing_Balance'] = 0.0

# Build balances
ratepension_balance = 0
aldersopsparing_balance = 0
ratepension_yearly_cost = initial_ratepension_yearly_cost

for idx, row in salary_data.iterrows():
    # Contributions
    ratepension_contribution = row['Ratepension_Contribution']
    aldersopsparing_contribution = row['Aldersopsparing_Contribution'] / avg_post_tax_rate  # gross up aldersopsparing

    # Apply investment returns after tax and deduct cost
    if idx > 0:
        ratepension_balance = (ratepension_balance - ratepension_yearly_cost) * (1 + real_investment_return * (1 - tax_on_gains))
        aldersopsparing_balance = aldersopsparing_balance * (1 + real_investment_return * (1 - tax_on_gains))

        # Inflate administration cost yearly
        ratepension_yearly_cost *= (1 + inflation_rate)

    # Add yearly contributions
    ratepension_balance += ratepension_contribution
    aldersopsparing_balance += aldersopsparing_contribution

    # Update balances in DataFrame
    salary_data.at[idx, 'Ratepension_Balance'] = ratepension_balance
    salary_data.at[idx, 'Aldersopsparing_Balance'] = aldersopsparing_balance

# Retirement phase (after accumulation)
retirement_start_year = salary_data['Year'].max() + 1
retirement_years = 25

# Last known balances
final_ratepension_balance = ratepension_balance
final_aldersopsparing_balance = aldersopsparing_balance

# Retirement withdrawals
ratepension_yearly_income = final_ratepension_balance / retirement_years
aldersopsparing_lump_sum = final_aldersopsparing_balance  # taken at first year of retirement

# Build retirement DataFrame
retirement_data = {
    'Year': [],
    'Ratepension_Income': [],
    'Aldersopsparing_Withdrawal': []
}

for i in range(retirement_years):
    year = retirement_start_year + i
    ratepension_income = ratepension_yearly_income * ((1 + inflation_rate) ** i)
    aldersopsparing_withdrawal = aldersopsparing_lump_sum if i == 0 else 0

    retirement_data['Year'].append(year)
    retirement_data['Ratepension_Income'].append(ratepension_income)
    retirement_data['Aldersopsparing_Withdrawal'].append(aldersopsparing_withdrawal)

retirement_df = pd.DataFrame(retirement_data)

#%%
print(salary_data)
#%%
print(retirement_df)
#%%
salary_data.to_csv(os.path.join(output_folder, "pension_contributionP1.csv"))
#%%
retirement_df.to_csv(os.path.join(output_folder, "retirement_infoP1.csv"))