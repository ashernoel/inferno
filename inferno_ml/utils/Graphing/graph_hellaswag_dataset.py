import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline (Dense)', 'Skewness (Hellaswag)', 'L2 Norm (Hellaswag)',
          'Magnitude', 'SparseGPT', 'Wanda']

accuracies = [57.2, 53.6, 52.41, 49.29, 52.37, 52.41]

# Colors
colors = ['red', 'blue', 'lightblue', 'green', 'limegreen', 'darkgreen']

# Calculate averages for the horizontal lines
avg_ours = np.mean([accuracies[models.index('Skewness (Hellaswag)')]])
avg_sota = np.mean([accuracies[models.index('Wanda')]])

# Create bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.title('Various models at 50% Sparsity on the Hellaswag dataset')
plt.xticks(rotation=45)  # rotate x-axis labels for better readability
plt.ylim(45, 65)  # you can modify this if you want to change the y-axis range

# Add horizontal lines and labels
plt.axhline(y=avg_ours, color='c', linestyle='dashed', linewidth=1)
plt.text(len(models)-1, avg_ours, ' Ours \n', verticalalignment='center', color='c')

plt.axhline(y=avg_sota, color='m', linestyle='dashed', linewidth=1)
plt.text(len(models)-1, avg_sota, ' SOTA \n', verticalalignment='center', color='g')

# Display the graph
plt.tight_layout()  # adjusts the plot to ensure everything fits without overlapping
plt.show()

plt.savefig("hellaswag.png")
