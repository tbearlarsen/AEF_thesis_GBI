import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load your CSV
profile = "P2"
df = pd.read_csv(
    rf'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\{profile}\goal_redemption_details.csv')
plot_dir = Path(rf"C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Runs\Base\{profile}\graphs")
plot_dir.mkdir(parents=True, exist_ok=True)

# Ensure 'Status' is boolean
df['Status'] = df['Status'].astype(str).str.upper() == 'TRUE'
df['Redeemed %'] = df['Redeemed Amount (DKK)'] / df['Goal Value (DKK)']
goal_names = df['Goal'].unique()


# --- Redemption Category Plot ---
def categorize_redemption(pct):
    if pct == 1.0:
        return 'Full (100%)'
    elif pct == 0.0:
        return 'None (0%)'
    else:
        return 'Partial'


for goal in goal_names:
    goal_data = df[df['Goal'] == goal]
    redeemed_by_path = goal_data.groupby('Path_ID')['Redeemed %'].max()
    categories = redeemed_by_path.apply(categorize_redemption).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    categories.reindex(['None (0%)', 'Partial', 'Full (100%)']).plot(kind='bar', ax=ax, edgecolor='black',
                                                                     color='#4a90e2')
    ax.set_title(f"Goal {goal} Redemption Status Distribution", fontsize=14)
    ax.set_xlabel("Redemption Category", fontsize=12)
    ax.set_ylabel("Number of Paths", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(plot_dir / f"{profile}_{goal}_result.png")
    plt.close()

# --- Partial Redemption Distribution ---
for goal in goal_names:
    goal_data = df[df['Goal'] == goal]
    redeemed_by_path = goal_data.groupby('Path_ID')['Redeemed %'].max()
    partials = redeemed_by_path[(redeemed_by_path > 0.0) & (redeemed_by_path < 1.0)]

    if not partials.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(partials, bins=15, edgecolor='black', color='#f5a623', density=True)
        ax.set_title(f"Distribution of Partial Redemptions – Goal {goal}", fontsize=14)
        ax.set_xlabel("Redeemed % of Goal Value (Partial Cases Only)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{profile}_{goal}_partial.png")
        plt.close()
    else:
        print(f"No partial redemptions for Goal {goal}")
