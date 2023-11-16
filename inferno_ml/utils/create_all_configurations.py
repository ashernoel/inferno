import csv
import torch
import SE_Swish as se_swish
from ptflops import get_model_complexity_info

# Define a function to compute FLOPs for a given configuration
def compute_flops(block, depth, width):
    with torch.cuda.device(0):
        net = se_swish.MicroCNN(block, depth, width, 0, 5)
        flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)

        # Delete the model and clear the GPU memory
        del net
        torch.cuda.empty_cache()
    return flops

# Prepare a list to store the rows
data_rows = []

print("About to start for loop")
for block in range(1, 8):  # Block count from 1 to 6
    for depth in range(1, 16):  # Depth from 1 to 9
        for width in range(1, 8):  # Width from 1 to 6
            flops = compute_flops(block, depth, width)
            if flops <= 5e9:  # Exclude configurations where FLOPs is over 5e9
                data_rows.append([block, depth, width, flops])
                print(f"Block: {block}, Depth: {depth}, Width: {width}, FLOPs: {flops}.")

# Sort data rows by FLOPs
data_rows.sort(key=lambda x: x[3])

# Write sorted rows to CSV
with open('configurations_filtered_sorted_flops.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Block', 'Depth', 'Width', 'FLOPs'])  # Write header
    csvwriter.writerows(data_rows)  # Write data rows

print("CSV generation complete.")
