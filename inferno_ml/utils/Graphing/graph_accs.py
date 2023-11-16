import matplotlib.pyplot as plt
import numpy as np
print("got here")

# Channel and kernel sizes
cnn1_channels_kernels = [(64, 3), (64, 1), (16, 3), (64, 1), (64, 1)]
cnn2_channels_kernels = [(64, 7), (64, 3), (64, 1), (64, 1), (0, 0)]

width = 0.35
# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
x = np.arange(len(cnn2_channels_kernels))
print("got here")

# Plot the Optimized CNN on the first subplot
ax1.bar(x - width/2, [ck[0] for ck in cnn1_channels_kernels], width, color='grey', label='Channels')

# Create a secondary y-axis for the first subplot
ax1_kernel = ax1.twinx()
ax1_kernel.bar(x + width/2, [ck[1] for ck in cnn1_channels_kernels], width, color='black', label='Kernels')

ax1.set_xticks(range(len(cnn1_channels_kernels)))
ax1.set_xticklabels(['Conv1', 'Residual: Conv2', 'Residual: Conv3', 'Residual: Conv4', 'Residual: Conv5'])
ax1.set_ylabel('Channels')
ax1_kernel.set_ylabel('Kernel Sizes')
ax1.set_title('Optimized CNN')
ax1.legend(loc='upper left')
ax1_kernel.legend(loc='upper right')

# Plot the Baseline CNN on the second subplot
ax2.bar(x - width/2, [ck[0] for ck in cnn2_channels_kernels], width, color='gray', label='Channels')
print("got here")
# Create a secondary y-axis for the second subplot
ax2_kernel = ax2.twinx()
ax2_kernel.bar(x + width/2, [ck[1] for ck in cnn2_channels_kernels], width, color='black', label='Kernels')

ax2.set_xticks(range(len(cnn2_channels_kernels)))
ax2.set_xticklabels(['Conv1', 'Residual: Conv2', 'Residual: Conv3', 'Residual: Conv4', 'Residual: Conv5'])
ax2.set_ylabel('Channels')
ax2_kernel.set_ylabel('Kernel Sizes')
ax2.set_title('Baseline CNN')
ax2.legend(loc='upper left')
ax2_kernel.legend(loc='upper right')
print("got here")

# Save the subplots to disk
plt.savefig('cnn_graphs1.png')

# Show the subplots
plt.show()
