import pandas as pd
import matplotlib.pyplot as plt

def family(row):
    if row["Width"] > 3*row["Depth"] and row["Width"] > row["Block"]:
        return "red"
    elif row["Width"] > 2*row["Depth"] and row["Width"] > row["Block"]:
        return "orange"
    elif row["Depth"] > row["Width"] and row["Depth"] > row["Block"]:
        return "blue"
    elif row["Width"] > row["Depth"] and row["Width"] > row["Block"]:
        return "yellow"
    else:
        return "green"

# Read the CSV file
csv_file = "configurations_max_blocks_5_max_depth_15_max_width_15.csv"
df = pd.read_csv(csv_file)


df = df[df["accuracy3"] != 0]

# Create a new column "Cluster_Label" that starts from 1 and increments sequentially
df["Cluster_Label"] = df["Cluster"].astype("category").cat.codes + 1

# Calculate the average FLOP value for each cluster
average_flops = df.groupby("Cluster_Label")["FLOPs"].mean()

# Set the style and theme for the graph
plt.style.use("seaborn-whitegrid")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.53
plt.rcParams["figure.figsize"] = (10, 6)

# Map the average FLOPs for each cluster to the data points
df["Average_FLOPs"] = df["Cluster_Label"].map(average_flops)

# Assign a family based on the model configurations
df["Family"] = df.apply(family, axis=1)

# Create the scatter plot
scatter = plt.scatter(df["Average_FLOPs"], df["accuracy3"], c=df["Family"], edgecolors="k", s=100, alpha=0.8)

# Customize the graph
plt.title("Average FLOPs vs Accuracy", fontsize=18, fontweight="bold")
plt.xlabel("Average FLOPs", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Create a legend with cluster labels and family descriptions
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", label="Width > 3 * Depth, Block", markerfacecolor="red", markersize=10, markeredgecolor="k"),
        plt.Line2D([0], [0], marker="o", color="w", label="Width > 2 * Depth, Block", markerfacecolor="orange", markersize=10, markeredgecolor="k"),
    plt.Line2D([0], [0], marker="o", color="w", label="Width > Depth, Block", markerfacecolor="yellow", markersize=10, markeredgecolor="k"),

    plt.Line2D([0], [0], marker="o", color="w", label="Depth > Width, Block", markerfacecolor="blue", markersize=10, markeredgecolor="k"),
    plt.Line2D([0], [0], marker="o", color="w", label="Block > Width, Depth", markerfacecolor="green", markersize=10, markeredgecolor="k"),
]

plt.legend(handles=legend_elements, fontsize=12, loc="best")

# Create a colorbar for the scatter plot
# cbar = plt.colorbar(scatter)
# cbar.ax.get_yaxis().labelpad = 15
# cbar.ax.set_ylabel("Cluster #", rotation=270, fontsize=14, fontweight="bold")

# Save the graph as an image file
plt.savefig("average_flops_vs_accuracy4.png", dpi=300)

# Show the graph
plt.show()