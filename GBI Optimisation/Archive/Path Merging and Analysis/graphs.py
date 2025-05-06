import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.patches as patches

# --- Function to draw gauge chart ---
def draw_gauge_chart(value, title, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(4, 4))  # square aspect to avoid stretching
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Smooth gradient from red to yellow to green
    cmap = LinearSegmentedColormap.from_list("custom", ["#e74c3c", "#f1c40f", "#2ecc71"], N=300)

    # Draw gradient arcs
    for i in range(100):
        theta1 = 180 - (i + 1) * 1.8
        theta2 = 180 - i * 1.8
        arc = patches.Wedge(center=(0, 0), r=1, theta1=theta1, theta2=theta2,
                            width=0.15, facecolor=cmap(i / 100))
        ax.add_patch(arc)

    # Draw needle
    angle = 180 - value * 100 * 1.8
    rad = np.radians(angle)
    x = 0.85 * np.cos(rad)
    y = 0.85 * np.sin(rad)
    ax.plot([0, x], [0, y], color='black', linewidth=3)
    ax.plot(0, 0, 'o', color='black', markersize=6)

    # Labels
    for pct in [0, 25, 50, 75, 100]:
        angle = 180 - pct * 1.8
        rad = np.radians(angle)
        x = 1.1 * np.cos(rad)
        y = 1.1 * np.sin(rad)
        ax.text(x, y, f"{pct}%", ha='center', va='center', fontsize=9)

    # Center value
    ax.text(0, -0.15, f"{value * 100:.0f}%", ha='center', va='center', fontsize=16, fontweight='bold')

    # Title
#    ax.set_title(title, fontsize=13, pad=20)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()

# --- Load CSV and setup ---
profile = "P1"
df = pd.read_csv(
    rf'C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Modelling\{profile}\goal_redemption_details.csv')
plot_dir = Path(rf"C:\Users\admin\CBS - Copenhagen Business School\Thesis - General\Modelling\{profile}\graphs")
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Data cleaning ---
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
        return 'Partial (1–99%)'

for goal in goal_names:
    goal_data = df[df['Goal'] == goal]
    redeemed_by_path = goal_data.groupby('Path_ID')['Redeemed %'].max()
    categories = redeemed_by_path.apply(categorize_redemption).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    categories.reindex(['None (0%)', 'Partial (1–99%)', 'Full (100%)']).plot(kind='bar', ax=ax, edgecolor='black', color='#4a90e2')
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

        bins = np.arange(0.0, 1.1, 0.1)
        counts, bins, patches_ = ax.hist(partials, bins=bins, edgecolor='black', color='#f5a623', density=False)

        ax.set_title(f"Distribution of Partial Redemptions – Goal {goal}", fontsize=14)
        ax.set_xlabel("Redeemed % of Goal Value (Partial Cases Only)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_xticklabels([f"{int(x * 100)}%" for x in np.arange(0.0, 1.1, 0.1)])

        for count, patch in zip(counts, patches_):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, count + 0.05, f"{int(count)}",
                        ha='center', va='bottom', fontsize=10)

        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{profile}_{goal}_partial.png")
        plt.close()
    else:
        print(f"No partial redemptions for Goal {goal}")

# --- Gauge Plot for % of fully redeemed paths ---
for goal in goal_names:
    goal_data = df[df['Goal'] == goal]
    redeemed_by_path = goal_data.groupby('Path_ID')['Redeemed %'].max()
    total_paths = len(redeemed_by_path)
    fully_redeemed = (redeemed_by_path == 1.0).sum()
    redemption_ratio = fully_redeemed / total_paths if total_paths > 0 else 0

    draw_gauge_chart(redemption_ratio, f"Fully Redeemed Paths – Goal {goal}", plot_dir / f"{profile}_{goal}_gauge.png")
