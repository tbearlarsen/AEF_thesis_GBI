import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from codelib.file_management.dynamic_file_pathing import get_root

def plot_percentage_distribution_for_goal(
    csv_path: str,
    goal_name,
    title: str = None
):
    """
    Load data from CSV, filter to a single Goal, compute achievement percentage,
    and plot the 1–100% distribution across all Monte Carlo paths (Path_ID).

    Parameters
    ----------
    csv_path : str
        Path to your 'goal_redemption_details.csv'.
    goal_name : same type as df['Goal']
        The specific value in the 'Goal' column you want to plot (e.g. 'A', 'B', etc.).
    title : str, optional
        Chart title. If None, defaults to 'Goal {goal_name} Achievement Distribution'.
    """
    # 1. Read the data
    df = pd.read_csv(csv_path)

    # 2. Filter to the specified goal
    subset = df[df['Goal'] == goal_name]
    if subset.empty:
        raise ValueError(f"No rows found for Goal = {goal_name!r}")

    # 3. Compute percentage achieved per Path_ID
    pct = (subset['Redeemed Amount (DKK)'] / subset['Goal Value (DKK)']) * 100

    # 4. Convert to integers 1–100
    arr = pct.values
    # if already 0–1, scale up
    if arr.min() >= 0 and arr.max() <= 1:
        arr = np.round(arr * 100)
    arr_int = np.clip(arr.astype(int), 1, 100)

    # 5. Count occurrences for each percentage
    counts = np.bincount(arr_int, minlength=101)[1:]  # index 1→100
    x = np.arange(1, 101)

    # 6. Plot
    plt.figure(figsize=(10, 5))
    plt.bar(x, counts)
    plt.xlabel('Percentage of Goal Achieved')
    plt.ylabel('Number of Simulation Paths')
    chart_title = title or f"Goal {goal_name} Achievement Distribution"
    plt.title(chart_title)
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(1, 100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example: plot for Goal 'A'
    plot_percentage_distribution_for_goal(
        'path/to/goal_redemption_details.csv',
        goal_name='A',
        title="Monte Carlo: Goal A Achievement"
    )
