import matplotlib.pyplot as plt
import numpy as np

def autolabel(ax, bars, xpos='center'):
    """
    Attach a text label above each bar in *bars*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()*offset[xpos], 1.01*height,
                '{:.0f}'.format(height), ha=ha[xpos], va='bottom')

# Data
cnn1_channels_kernels = [(64, 3), (64, 1), (16, 3), (64, 1), (64, 1)]
cnn2_channels_kernels = [(64, 7), (64, 3), (64, 1), (64, 1), (0, 0)]

width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
x = np.arange(len(cnn2_channels_kernels))

# Optimized CNN
bars1_channels = ax1.bar(x - width/2, [ck[0] for ck in cnn1_channels_kernels], width, color='darkgrey', label='Channels')
bars1_kernels = ax1.bar(x + width/2, [ck[1] for ck in cnn1_channels_kernels], width, color='black', label='Kernels')

# Baseline CNN
bars2_channels = ax2.bar(x - width/2, [ck[0] for ck in cnn2_channels_kernels], width, color='darkgrey', label='Channels')
bars2_kernels = ax2.bar(x + width/2, [ck[1] for ck in cnn2_channels_kernels], width, color='black', label='Kernels')

# Labels and titles
ax1.set_xticks(range(len(cnn1_channels_kernels)))
ax1.set_xticklabels(['Conv1', 'Residual: Conv2', 'Residual: Conv3', 'Residual: Conv4', 'Residual: Conv5'])
ax1.set_ylabel('Number')
ax1.set_title('Optimized CNN')
ax1.legend()

ax2.set_xticks(range(len(cnn2_channels_kernels)))
ax2.set_xticklabels(['Conv1', 'Residual: Conv2', 'Residual: Conv3', 'Residual: Conv4', 'Residual: Conv5'])
ax2.set_ylabel('Number')
ax2.set_title('Baseline CNN')
ax2.legend()
ax1.set_ylim([0, 70])
ax2.set_ylim([0, 70])
# Add labels above each bar
autolabel(ax1, bars1_channels, 'center')
autolabel(ax1, bars1_kernels, 'center')
autolabel(ax2, bars2_channels, 'center')
autolabel(ax2, bars2_kernels, 'center')
# Save the subplots to disk
plt.savefig('cnn_graphs.png')

# Show the subplots
plt.show()
