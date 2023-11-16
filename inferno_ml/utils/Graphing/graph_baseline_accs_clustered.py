import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load data
df = pd.read_csv('configurations_ALL_flops.csv')

# Select features to perform clustering
df = df[df['accuracy_imagenet5'] > 0.8]
df = df[df['FLOPs'] < 5e9]
df = df.reset_index(drop=True)  # Resetting the index


nnet_calculator_data = [
    {"k": 7, "d": 3, "e": 6, "FLOPs": 677.830512, "Accuracy": 82.6923076923077},
    {"k": 7, "d": 4, "e": 6, "FLOPs": 892.671792, "Accuracy": 86.92307692307692},
]
nnet_calculator_df = pd.DataFrame(nnet_calculator_data)
nnet_calculator_data2 = [
    {"k": 7, "d": 3, "e": 6, "FLOPs": 820.830512, "Accuracy": 80.0},
    {"k": 7, "d": 4, "e": 6, "FLOPs": 1058.671792, "Accuracy": 81.92},
]
nnet_calculator_df2 = pd.DataFrame(nnet_calculator_data2)


features = df[['Block', 'Depth', 'Width']]

# Set number of clusters
n_clusters = 6

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(features)

# Add cluster assignments to the dataframe
df['cluster'] = kmeans.labels_

# Set up colors map
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Extract the cluster centers
cluster_centers = kmeans.cluster_centers_

# Plot each cluster
plt.figure(figsize=(10,6))
plt.title('K-Means Clustering of Configurations', fontsize=18)
for i in range(n_clusters):
    cluster_data = df[df['cluster'] == i]
    centroid_coords = [f"{coord:.2f}" for coord in cluster_centers[i]]
    centroid_label = f"Cluster {i + 1} - Centroid (Block, Depth, Width): ({', '.join(centroid_coords)})"
    plt.scatter(cluster_data['FLOPs'], cluster_data['accuracy_imagenet5'], color=colors[i], label=centroid_label)

# Label axes
plt.xlabel('FLOPs', fontsize=16)
plt.ylabel('accuracy_imagenet5', fontsize=16)

# Add legend
plt.legend(fontsize=14)

# Show the plot
plt.show()
plt.savefig("graph_all_imagenet5_baseline_with_centroids_new_CSV.png")
plt.close()

# Plotting the "ONCE FOR ALL MOBILE NET" data
plt.scatter(nnet_calculator_df['FLOPs']*1e6, nnet_calculator_df['Accuracy']/100.00, color='red', marker='s')
plt.plot(nnet_calculator_df['FLOPs']*1e6, nnet_calculator_df['Accuracy']/100.00, color='red', label='Once-For-All MobileNet')
plt.scatter(nnet_calculator_df2['FLOPs']*1e6, nnet_calculator_df2['Accuracy']/100.00, color='orange', marker='s')
plt.plot(nnet_calculator_df2['FLOPs']*1e6, nnet_calculator_df2['Accuracy']/100.00, color='orange', label='Once-For-All ProxylessNAS' )

def pareto_frontier(x, y, maxX = True, maxY = True):
    '''
    Function to find the pareto frontier from a set of points.
    :param x: x-values of the points
    :param y: y-values of the points
    :param maxX: Indicator whether to maximize x-values
    :param maxY: Indicator whether to maximize y-values
    :return: Sorted list of points forming the pareto frontier
    '''
    # Sort the points
    sorted_points = sorted([[x[i], y[i]] for i in range(len(x))], reverse=maxX)
    
    # Initialize the pareto frontier with the first point
    pareto_front = [sorted_points[0]]
    
    # Check each point if it improves the pareto frontier
    for point in sorted_points[1:]:
        if maxY: 
            if point[1] >= pareto_front[-1][1]:
                pareto_front.append(point)
        else:
            if point[1] <= pareto_front[-1][1]:
                pareto_front.append(point)
    
    # Extract the x and y values
    pareto_front = [(point[0], point[1]) for point in pareto_front]
    
    return pareto_front

# Extract the pareto frontier data
pareto_front = pareto_frontier(df['FLOPs'], df['accuracy_imagenet5'], maxX=False, maxY=True)
pareto_front_df = pd.DataFrame(pareto_front, columns=['FLOPs', 'accuracy_imagenet5'])
max_flop_pareto = pareto_front_df['FLOPs'].max()
plt.plot(pareto_front_df['FLOPs'], pareto_front_df['accuracy_imagenet5'], color='grey', label='NanoCNN Pareto Frontier')
filtered_df = df[df['FLOPs'] < max_flop_pareto]
plt.scatter(filtered_df['FLOPs'], filtered_df['accuracy_imagenet5'], color='black')


# Label axes
plt.xlabel('FLOPs', fontsize=14)
plt.ylabel('Top-1 Imagenet-5 Accuracy', fontsize=14)

# Add legend
plt.legend(fontsize=11, loc='upper right')
# Show the plot
# set y axis from 0.78 to 0.95
plt.ylim(0.79, 0.96)
plt.show()
plt.title("OFA vs. NanoCNN on Imagenet-5", fontsize=16)
plt.savefig("graph_all_imagenet5_baseline_with_additional_data_and_pareto_frontier.png")
plt.close()


# New section for second plot
block_colors = {1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'm', 6: 'y'}
plt.figure(figsize=(10,6))
plt.title('Configuration Distribution by Block', fontsize=18)
for block in block_colors.keys():
    block_data = df[df['Block'] == block]
    plt.scatter(block_data['FLOPs'], block_data['accuracy_imagenet5'], color=block_colors[block], label=f'Block {block}')

plt.xlabel('FLOPs', fontsize=16)
plt.ylabel('accuracy_imagenet5', fontsize=16)
plt.legend(fontsize=14)
plt.show()
plt.savefig("graph_all_imagenet5_baseline_with_blocks_new_CSV_2.png")
plt.close()




block_colors = {1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'm', 6: 'y'}
plt.figure(figsize=(10,6))
plt.title('Configuration Distribution by Width', fontsize=18)
for width in block_colors.keys():
    width_data = df[df['Width'] == width]
    plt.scatter(width_data['FLOPs'], width_data['accuracy_imagenet5'], color=block_colors[width], label=f'Width {width}')

plt.xlabel('FLOPs', fontsize=16)
plt.ylabel('accuracy_imagenet5', fontsize=16)
plt.legend(fontsize=14)
plt.show()
plt.savefig("graph_all_imagenet5_baseline_with_blocks_new_CSV_4.png")
plt.close()





# New section for third plot (zooming into Cluster 4 and reclustering)
cluster_4_data = df[df['cluster'] == 3].copy()  # Creating a copy to avoid SettingWithCopyWarning
features_cluster_4 = cluster_4_data[['Block', 'Depth', 'Width']]
n_subclusters = 5
kmeans_cluster_4 = KMeans(n_clusters=n_subclusters, n_init=10, random_state=0).fit(features_cluster_4)
cluster_4_data['subcluster'] = kmeans_cluster_4.labels_
subcluster_centers = kmeans_cluster_4.cluster_centers_
plt.figure(figsize=(10,6))
plt.title('Re-Clustering within Cluster 4', fontsize=18)
for i in range(n_subclusters):
    subcluster_data = cluster_4_data[cluster_4_data['subcluster'] == i]
    centroid_coords = [f"{coord:.2f}" for coord in subcluster_centers[i]]
    centroid_label = f"Subcluster {i + 1} - Centroid (Block, Depth, Width): ({', '.join(centroid_coords)})"
    plt.scatter(subcluster_data['FLOPs'], subcluster_data['accuracy_imagenet5'], color=colors[i], label=centroid_label)

plt.xlabel('FLOPs', fontsize=16)
plt.ylabel('accuracy_imagenet5', fontsize=16)
plt.legend(fontsize=14)
plt.show()
plt.savefig("graph_all_imagenet5_baseline_with_blocks_new_CSV_3.png")

