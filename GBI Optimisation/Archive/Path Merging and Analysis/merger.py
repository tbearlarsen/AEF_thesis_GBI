import os
import pandas as pd

# List your folder paths here


folders = [
    r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1\outputs_path_0_to_24',
    r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1\outputs_path_25_to_50',
    r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1\outputs_path_51_to_75',
    r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1\outputs_path_76_100',
    r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1\outputs_path_101_to_200'
    # add as many as needed
]

# List of filenames to look for (from your image)
file_names = [
    'goal_redemption_details.csv',
    'goal_success_summary.csv',
    'master_aggregate_portfolio_by_year.csv',
    'master_asset_returns_log.csv',
    'master_final_wealth_by_year.csv',
    'master_goal_weights_by_year.csv',
    'master_optimal_weights_goal_A.csv',
    'master_optimal_weights_goal_B.csv',
    'master_optimal_weights_goal_C.csv',
    'master_optimal_weights_goal_D.csv',
]

# Output folder
output_dir = r'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\P1'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each file name and combine data
for file_name in file_names:
    combined_df = pd.DataFrame()
    for folder in folders:
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['source_folder'] = os.path.basename(folder)  # Optional: add a column to track origin
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    # Save combined DataFrame
    output_path = os.path.join(output_dir, file_name)
    combined_df.to_csv(output_path, index=False)
