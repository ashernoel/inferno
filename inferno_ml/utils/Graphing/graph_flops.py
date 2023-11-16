import matplotlib.pyplot as plt
import numpy as np

# Flop counts for each layer
cnn1_flops = [15.02, 17.1, 0.325]
cnn2_flops = [158.33, 320.86, 0.325]

cnn1_channels = [64, 64, 16, 64, 64, 5]
cnn2_channels = [64, 64, 64, 64, 5]

# Kernel size data
cnn1_kernel_sizes = [3, 1, 3, 1, 1, None]
cnn2_kernel_sizes = [7, 3, 1, 1, None]

# Layer labels
layer_labels = ['CNN', 'Residual Block', 'Linear']

# Set up bar graph
x = np.arange(len(cnn1_flops))
width = 0.3

fig, ax = plt.subplots()
bar1 = ax.bar(x - width, cnn1_flops, width, label='Optimized CNN', color='grey')
bar2 = ax.bar(x, cnn2_flops, width, label='Baseline CNN', color='black')

# Set labels and legend
ax.set_ylabel('Flop Count (MMac)')
ax.set_title('Flop Count for Two (1 block count, 1 depth, 1 width) CNNs')
ax.set_xticks(np.arange(3) - width / 2)  # Move x-axis ticks to the midpoint of the bars
ax.set_xticklabels(layer_labels)
ax.legend()

# Function to auto-label bars with their values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)

# Increase the height of the y-axis
ax.set_ylim(0, 350)

plt.show()
fig.savefig(f'Flops vs. layer.png')
