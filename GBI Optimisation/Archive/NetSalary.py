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
excel_path = os.path.join(data_folder, "salaryP2 Nominal Inflation Adjusted.xlsx")

# Load your salary data
salary_data = pd.read_excel(excel_path)  # Adjust path if needed

# Rename columns for clarity
salary_data.columns = ['Year', 'Gross_Salary']
#%%
# Define constants
pension_rate = 0.05  # 5% pension contribution
am_bidrag_rate = 0.08  # 8% labour market contribution
bottom_state_tax_rate = 0.1215  # 12.15% bottom state tax
municipal_tax_rate = 0.25  # 25% municipal tax average
top_state_tax_rate = 0.15  # 15% top state tax
top_tax_threshold = 618400  # 2025 top tax threshold (in DKK)
#%%
# Function to calculate post-tax salary precisely
def calculate_post_tax_salary(gross_salary):
    # Step 1: Deduct pension
    salary_after_pension = gross_salary * (1 - pension_rate)

    # Step 2: Deduct AM-bidrag
    salary_after_am = salary_after_pension * (1 - am_bidrag_rate)

    # Step 3: Apply bottom state tax and municipal tax
    bottom_state_tax = salary_after_am * bottom_state_tax_rate
    municipal_tax = salary_after_am * municipal_tax_rate

    # Step 4: Apply top state tax if applicable
    if salary_after_am > top_tax_threshold:
        top_state_tax = (salary_after_am - top_tax_threshold) * top_state_tax_rate
    else:
        top_state_tax = 0

    # Step 5: Calculate total tax and net salary
    total_tax = bottom_state_tax + municipal_tax + top_state_tax
    post_tax_salary = salary_after_am - total_tax

    return post_tax_salary

# Apply the function
salary_data['Post_Tax_Salary'] = salary_data['Gross_Salary'].apply(calculate_post_tax_salary)
#%%
salary_data.to_csv(os.path.join(output_folder, "net_salaryP2.csv"))