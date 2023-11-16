import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import inferno_ml.utils.NanoCNN as se_swish
from sklearn.cluster import KMeans
import numpy as np
import math
from collections import defaultdict
import os, psutil
import concurrent.futures
import traceback
import csv

def format_flops(flops):
    if flops == float("inf"):
        return "inf"
    if flops == float("-inf"):
        return "-inf"
    if flops <= 0:
        return "N/A"
    exponent = int(math.log10(flops))
    mantissa = flops / 10 ** exponent
    return f"{mantissa:.3g}e{exponent}"

def is_tuple_close_to_own_centroid(t_flops, centroids, min_distance_ratio = 1.5):
    own_cluster = label_mapping[kmeans.predict(np.array([[t_flops]]))[0]]
    own_centroid = centroids[own_cluster][1]
    other_centroids = [c for i, c in enumerate(centroids) if i != own_cluster]

    own_distance = abs(t_flops - own_centroid)
    min_other_distance = min([abs(t_flops - c) for _, c in other_centroids])

    return own_distance * min_distance_ratio <= min_other_distance


def save_results_to_csv(flop_list, max_blocks, max_depth, max_width, residual, filename=None):
    if filename is None:
        filename = f'results_blocks_{max_blocks}_depth_{max_depth}_width_{max_width}_residual_{residual}.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Block', 'Depth', 'Width', 'Residual', 'FLOPs', 'Cluster'])
        print(flop_list)
        for b, d, w, r, flops, cluster, _ in flop_list:
            formatted_flops = format_flops(flops)
            csvwriter.writerow([b, d, w, r, formatted_flops, cluster])

def evaluate_model_on_gpu(gpu_id, block, depth, width, residual):
    with torch.cuda.device(gpu_id):
        net = se_swish.MicroCNN(block, depth, width, residual, 5)
        flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)

        # Delete the model and clear the GPU memory
        del net
        torch.cuda.empty_cache()

    return flops

def parallel_evaluate_model(args):
    try:
        gpu_id, block, depth, width, residual = args
        return block, depth, width, residual, evaluate_model_on_gpu(gpu_id, block, depth, width, residual)
    except Exception as e:
        print(f"Exception in parallel_evaluate_model: {e}")
        traceback.print_exc()
        raise

def generate_cluster_boundaries(start, first_end, total_clusters, final_boundary):
    cluster_boundaries = [start, first_end]
    remaining_range = final_boundary - first_end
    total_intervals = total_clusters - 1
    sum_intervals = (total_intervals * (total_intervals + 1)) // 2
    base_interval = remaining_range / sum_intervals

    current_boundary = first_end
    for i in range(2, total_clusters):
        increment = base_interval * i
        current_boundary += increment
        cluster_boundaries.append(current_boundary)
    cluster_boundaries[-1] = final_boundary
    return cluster_boundaries

def calculate_midpoints(cluster_boundaries):
    centroids = {}
    for i in range(len(cluster_boundaries) - 1):
        midpoint = (cluster_boundaries[i] + cluster_boundaries[i + 1]) / 2
        centroids[i] = midpoint
    return centroids

def generate_flop_list(max_blocks, max_depth, max_width, residual):
    num_gpus = torch.cuda.device_count()

    all_configs = []

    for block in range(1, max_blocks + 1):
        depth = 1
        while depth <= max_depth:
            width = 1
            while width <= max_width:       
                all_configs.append((block, depth, width, residual))
                width += 1
            depth += 1

    flop_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        i = 0
        while i < len(all_configs):
            gpu_id = i % num_gpus
            args = (gpu_id,) + all_configs[i]
            future = executor.submit(parallel_evaluate_model, args)
            result = future.result()

            b, d, w, r, flops = result

            if flops <= 5e9:
                flop_list.append((b, d, w, r, flops))
                i += 1
            else:
                i += max_width - w + 1

    start_boundary = 30e6
    first_end_boundary = 70e6
    num_clusters = 16
    final_boundary = 5e9

    cluster_boundaries = generate_cluster_boundaries(start_boundary, first_end_boundary, num_clusters, final_boundary)
    print(cluster_boundaries)


    # Assign each model to the appropriate cluster based on its FLOPs
    flop_array = [(b, d, w, r, f, sum(f > boundary for boundary in cluster_boundaries) - 1) for b, d, w, r, f in flop_list]

    
    centroids = calculate_midpoints(cluster_boundaries)

    # Create a mapping of old cluster labels to new cluster labels based on the FLOPs
    label_mapping = {}
    for new_label, old_label in enumerate(centroids):
        label_mapping[old_label] = new_label

    # Assign new cluster labels to the tuples
    flop_array = [(b, d, w, r, f, label_mapping[cluster]) for b, d, w, r, f, cluster in flop_array]


        # Sort flop_array based on cluster, then by the priority order
    flop_array.sort(key=lambda x: (x[5], -x[1], -x[2], -x[0], x[1] == x[2]))

    # Initialize dictionaries to store the largest depth and width tuples for each cluster
    largest_depth_tuples = defaultdict(lambda: (None, float('-inf')))
    largest_width_tuples = defaultdict(lambda: (None, float('-inf')))

    # Initialize dictionaries to store if the max depth and max width conditions are met for each cluster
    max_depth_met = defaultdict(bool)
    max_width_met = defaultdict(bool)

    # Initialize dictionaries to store the minimum and maximum FLOPs for each printed cluster
    min_flops_printed = defaultdict(lambda: float("inf"))
    max_flops_printed = defaultdict(lambda: float("-inf"))

    # filtered_flop_array = [t for t in flop_array if t[4] <= cluster_boundaries[t[5]]]

    # Find the largest depth and width tuples for each cluster
    for b, d, w, r, f, cluster in flop_array:
        if d > largest_depth_tuples[cluster][1]:
            largest_depth_tuples[cluster] = ((b, d, w, r, f, cluster), d)
        if w > largest_width_tuples[cluster][1]:
            largest_width_tuples[cluster] = ((b, d, w, r, f, cluster), w)

    printed_tuples = []
    printed_clusters = defaultdict(lambda: [False, False])  # Track if the residual value conditions are met
    for b, d, w, r, f, cluster in flop_array:
        reason = ""
        if (b, d, w, r, f, cluster) == largest_depth_tuples[cluster][0]:
            reason = "max depth"
        elif (b, d, w, r, f, cluster) == largest_width_tuples[cluster][0]:
            reason = "max width"
        elif d == w:
            reason = "equal depth & width"

        if reason:  # If there's a reason to print the tuple, do so
            printed_tuples.append((b, d, w, r, f, cluster, reason))

            # Update the minimum and maximum FLOPs for the printed cluster
            if f < min_flops_printed[cluster]:
                min_flops_printed[cluster] = f
            if f > max_flops_printed[cluster]:
                max_flops_printed[cluster] = f

    save_results_to_csv(printed_tuples, max_blocks, max_depth, max_width, residual)
    return printed_tuples