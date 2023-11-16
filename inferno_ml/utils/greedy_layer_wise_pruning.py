import torch
import csv
import os
import torch.nn as nn
from . import NanoCNN as se_swish
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from torch.cuda import device
import pandas as pd
from inferno_ml.utils.layer_wise_pruning import create_model_from_config, load_results_from_csv, transfer_weights, evaluate_model_on_gpu
from inferno_ml.utils.gpu_training_script import get_mean_and_std, train_model, get_data_transforms, optimizer_to, ImageFolderWithPaths

def plot_max_accuracies(ax, results, desired_width, alpha=0.5, linestyle="--"):
    """
    Plot the maximum accuracies for given results, considering only the configurations
    that are part of the Pareto frontier (i.e., configurations that are not dominated by
    any other in terms of both accuracy and FLOPs).
    """

    # Create lists to hold all accuracies, FLOPs and the same for the desired width
    all_accuracies = []
    all_flops = []
    width_accuracies = []
    width_flops = []

    # Go through each row in the results and extract the accuracy, FLOPs, and width
    for row in results:
        accuracy = float(row[4])  # Assuming the accuracy is in the 5th column
        flops = float(row[3])  # Assuming the FLOPs are in the 4th column
        width = float(row[2])  # Assuming the width is in the 3rd column

        # Append the accuracy and FLOPs to the respective lists
        all_accuracies.append(accuracy)
        all_flops.append(flops)

        # If the width matches the desired width, append to the width-specific lists
        if width == desired_width:
            width_accuracies.append(accuracy)
            width_flops.append(flops)

    # Combine FLOPs and accuracies into tuples for sorting
    all_points = list(zip(all_flops, all_accuracies))
    width_points = list(zip(width_flops, width_accuracies))

    # Sort points by FLOPs (this helps in finding the Pareto frontier)
    all_points.sort(key=lambda x: x[0])
    width_points.sort(key=lambda x: x[0])

    # Function to find the Pareto frontier
    def pareto_frontier(points):
        # List to store the Pareto frontier points
        pareto_points = []
        max_acc = float('-inf')

        for flops, acc in points:
            # If this point has a higher accuracy than the current max, add it to the frontier
            if acc > max_acc:
                pareto_points.append((flops, acc))
                max_acc = acc

        return pareto_points

    # Get the Pareto frontier for all points and for the desired width
    pareto_all = pareto_frontier(all_points)
    pareto_width = pareto_frontier(width_points)

    # Unzip the points into separate lists for plotting
    pareto_all_flops, pareto_all_accuracies = zip(*pareto_all)
    pareto_width_flops, pareto_width_accuracies = zip(*pareto_width)

    # Plot the Pareto frontier for all widths
    ax.plot(pareto_all_flops, pareto_all_accuracies, alpha=alpha, linestyle=linestyle, color="black", label="All Widths")
    # Plot the Pareto frontier for the desired width
    ax.plot(pareto_width_flops, pareto_width_accuracies, alpha=alpha, linestyle=linestyle, color="red", label=f"Width={desired_width}")




def evaluate_and_train_model(old_model, config, data_dir, num_classes, batch_size, input_size, mean, std, num_epochs, rank, num_gpus, device_string, criterion, scheduler, dataset_name):
    model = create_model_from_config(config, num_classes)
    model = transfer_weights(old_model, model)
    
    device = torch.device(device_string if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = f"{dataset_name}_block_{config[0]}_depth_{config[1]}_width_{config[2]}"


    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    data_transforms = get_data_transforms(input_size, mean, std)
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=num_gpus, rank=rank, shuffle=True, drop_last=False) for x in ['train', 'val']}
    image_dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2, drop_last=False, sampler=image_samplers[x], persistent_workers=True, prefetch_factor=2) for x in ['train', 'val']}

    model, val_acc = train_model(
        model=model,
        num_gpus=num_gpus,  
        data_dir=data_dir,  
        model_name=model_name, 
        rank=rank,
        dataloaders=image_dataloaders_dict,  
        criterion=criterion,
        optimizer=optimizer,  
        scheduler=scheduler, 
        input_size=input_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        start_epoch=0, 
        patience=10  
    )

    return val_acc, model


def plot_search_process(all_results, search_results, threshold_flops, total_epochs, config, config_end, filename):
    fig, ax = plt.subplots()

    
    # Sort the search results by the order they were added (if not already sorted)
    # search_results = sorted(search_results, key=lambda x: x[2] if len(x) > 2 else float('inf'))
    
    # Initialize the best accuracy to a very low number
    best_accuracy = -float('inf')
    prev_best_point = search_results[0]
    cur_best_point = None 

    # Plot each point and connect to the best previous point
    num = 0
    last_index = len(search_results) - 1  # Get the index of the last item

    for flops, accuracy in search_results:
        color = 'green' if num == 0 else ('darkblue' if num % 2 == 1 else 'lightblue')
        label = f'Start at {config}' if num == 0 else ('Block Decrement' if num % 2 == 1 else 'Depth Decrement')
        if num == last_index:
            color = 'red'
            label = f'End at {config_end}'

        ax.scatter(flops, accuracy, marker='o', color=color, label=label if num < 2 or (num % 2 == 0 and num < 4) else "")

        if num % 2 == 1:
            # Update the best accuracy and point
            best_accuracy = accuracy
            cur_best_point = (flops, accuracy)

        if num % 2 == 0:
            if accuracy > best_accuracy:
                cur_best_point = (flops, accuracy)
        
        if prev_best_point is not None:
            # Plot line from the current point to the best previous point
            ax.annotate("",
                        xy=(flops, accuracy), xycoords='data',
                        xytext=(prev_best_point[0], prev_best_point[1]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='grey'))
        
        if num % 2 == 0 and cur_best_point is not None:
            prev_best_point = cur_best_point
        num += 1
    
    # if all_results has any items:
    if len(all_results) > 0:
        plot_max_accuracies(ax, all_results, config[2])
    # Add vertical line for FLOP threshold
    ax.axvline(x=threshold_flops, color='red', linestyle='--', label='FLOP Threshold')

    configurations = pd.read_csv('configurations_ALL_flops.csv')
    configurations_under_1B = configurations[configurations['FLOPs'] < threshold_flops]
    average_accuracy_all_widths = configurations_under_1B['accuracy_imagenet5'].mean()
    average_accuracy_width_5 = configurations_under_1B[configurations_under_1B['Width'] == config[2]]['accuracy_imagenet5'].mean()
    ax.scatter(configurations_under_1B['FLOPs'], configurations_under_1B['accuracy_imagenet5'], alpha=0.5, facecolors='none', edgecolors='lightgrey', label='Configurations < 1B FLOPs')

    # Adding a horizontal line for average accuracy for all configurations under the threshold
    ax.axhline(y=average_accuracy_all_widths, color='red', linestyle='--', linewidth=1, label='Fully trained average accuracy (all widths)')

    # Adding a horizontal line for average accuracy for configurations under the threshold where width=5
    ax.axhline(y=average_accuracy_width_5, color='red', linestyle='-', linewidth=1, label='Fully trained average accuracy (width={})'.format(config[2]))

    ax.set_xlabel('FLOPs')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Greedy Search FLOPs vs Accuracy, total epochs: {total_epochs}')
    ax.legend()
    fig.savefig(filename)


def greedy_search(initial_config, threshold_flops, data_dir, num_classes, num_epochs, batch_size, input_size, mean, std, rank, num_gpus, device_string, criterion, scheduler, dataset_name):
    current_model = create_model_from_config(initial_config, num_classes)
    current_config = initial_config
   
    current_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{initial_config[0]}_depth_{initial_config[1]}_width_{initial_config[2]}_residual_False.pt"
    if not os.path.isfile(current_model_path):
        current_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{initial_config[0]}_depth_{initial_config[1]}_width_{initial_config[2]}_residual_0.pt"
    
    search_results = []  # Store tuples of (FLOPs, accuracy)
    all_results = []
    total_epochs = 0
    if os.path.isfile(current_model_path):
        current_model.load_state_dict(torch.load(current_model_path))
        # Call the function to plot the Pareto frontier
        csv_path = "configurations_ALL_flops.csv"
        all_results = load_results_from_csv(csv_path)
        # find the accuracay that corresponds to current_config in all_results
        index = 0 #UPDATE FOR DATASETS
        for row in all_results:
            width = int(row[2])
            block = int(row[0])
            depth = int(row[1])
            accuracy = float(row[4 + index])
            if block == initial_config[0] and depth == initial_config[1] and width == initial_config[2]:
                search_results.append((float(row[3]), accuracy))
                break
    # otherwise, train a model from scratch for 200 epochs 
    else:
        model = se_swish.MicroCNN(int(initial_config[0]), int(initial_config[1]), int(initial_config[2]), 0, num_classes)
        model = model.to(rank)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer_to(optimizer,torch.device(device_string))
        data_transforms = get_data_transforms(input_size,mean,std)
        model_name = f"{dataset_name}_block_{model_config[0]}_depth_{model_config[1]}_width_{model_config[2]}_residual_{model_config[3]}"
        image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=num_gpus, rank=local_rank, shuffle=True, drop_last=False) for x in ['train', 'val']}
        image_dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8, drop_last=False, sampler=image_samplers[x], persistent_workers=True, prefetch_factor=2) for x in ['train', 'val']}
        current_model, val_acc_history = train_model(model, num_gpus, data_dir, model_name, local_rank, image_dataloaders_dict, criterion, optimizer, None, input_size, batch_size, num_epochs=200, start_epoch=0, patience=10)
        search_results.append((evaluate_model_on_gpu(rank, *initial_config), val_acc_history[-1]))


    while True:
        # Decrement block count and depth
        # dont decrement depth or block count if it is already equal to one

        new_block_config = list(current_config)
        # dont decrement depth or block count if it is already equal to one
        if new_block_config[0] > 1:
            new_block_config[0] -= 1

        new_depth_config = list(current_config)
        if new_depth_config[1] > 1:
            new_depth_config[1] -= 1  # Decrement depth

        # Evaluate new configurations
        block_val_acc, block_model = evaluate_and_train_model(current_model, new_block_config, data_dir, num_classes, batch_size, input_size, mean, std, num_epochs, rank, num_gpus, device_string, criterion, scheduler, dataset_name)
        depth_val_acc, depth_model = evaluate_and_train_model(current_model, new_depth_config, data_dir, num_classes, batch_size, input_size, mean, std, num_epochs, rank, num_gpus, device_string, criterion, scheduler, dataset_name)

        # Get FLOPs for new configurations
        block_flops = evaluate_model_on_gpu(rank, *new_block_config)
        depth_flops = evaluate_model_on_gpu(rank, *new_depth_config)

        # Append results to search_results
        search_results.append((block_flops, block_val_acc))
        search_results.append((depth_flops, depth_val_acc))
        total_epochs += num_epochs*2
        # print everything associatd with the current iteration, including the configs, the accuracies,and the flops:
        print(f"Current config: {current_config}")
        print(f"Block config: {new_block_config}")
        print(f"Depth config: {new_depth_config}")
        print(f"Block val acc: {block_val_acc}")
        print(f"Depth val acc: {depth_val_acc}")
        print(f"Block flops: {block_flops}")
        print(f"Depth flops: {depth_flops}")

        # Choose the configuration with the higher validation accuracy
        if block_val_acc > depth_val_acc and block_flops >= threshold_flops:
            current_config = new_block_config
            current_model = block_model
            if current_config[0] == 1 and new_block_config[0] == new_depth_config[0]:
                print("Final config reached")
                break
            print("Block config chosen")
        elif depth_flops >= threshold_flops:
            current_config = new_depth_config
            current_model = depth_model
            if current_config[1] == 1 and new_block_config[1] == new_depth_config[1]:
                print("Final config reached")
                break
            print("Depth config chosen")
        else:
            if depth_val_acc > block_val_acc:
                current_config = new_depth_config
                current_model = depth_model
            else:
                current_config = new_block_config
                current_model = block_model
            print("FLOPs threshold reached here")
            break  # Break if both configurations are below the FLOPs threshold

        # Check if the current configuration is below the FLOPs threshold
        current_flops = evaluate_model_on_gpu(rank, *current_config)
        if current_flops < threshold_flops:
            print("FLOPs threshold reached")
            break

        ## if the current config has 1 block and 1 depth, then break
        if current_config[0] == 1 and current_config[1] == 1:
            print("Final config reached")
            break

    print("Greedy search complete. Starting 1 more epoch")
    # Continue training the final model for 50 more epochs
    final_epochs = num_epochs*31
    final_acc, final_model = evaluate_and_train_model(current_model, current_config, data_dir, num_classes, batch_size, input_size, mean, std, final_epochs, rank, num_gpus, device_string, criterion, scheduler, dataset_name)
    final_flops = evaluate_model_on_gpu(rank, *current_config)
    search_results.append((final_flops, final_acc))
    total_epochs += final_epochs

    plot_search_process(all_results, search_results, threshold_flops, total_epochs, initial_config, current_config, f'search_process_block_{initial_config[0]}_depth_{initial_config[1]}_width_{initial_config[2]}-{total_epochs}_epochs.png')

    return final_model, current_config

def main():

    initial_config = [5,5,4]
    threshold_flops = 1e9  # Define the FLOPs threshold
    
    data_dir = "/n/idreos_lab/users/usirin/datasets/imagenet_subsets/imagenet_training_5class_subset0_asher"
    dataset_name = "ic_imagenet_5class_subset0"
    # data_dir ="/n/idreos_lab/users/usirin/datasets/Multi-class-Weather-Dataset-for-Image-Classification/dataset2"
    # dataset_name = "weather"
    # data_dir = "/n/idreos_lab/users/usirin/datasets/blood-cell-images/dataset2-master/dataset2-master/images/TRAIN"
    # dataset_name = "bloodcell"
    num_classes = 0
    train_dir = data_dir + "/train"
    for filename in os.listdir(train_dir):
        num_classes = num_classes + 1
    batch_size=32
    rank=0
    device_string = "cuda:" + str(rank)
    num_gpus=1
    input_size = 256

    mean, std = get_mean_and_std(data_dir, batch_size, input_size)
    scheduler = None
    criterion = nn.CrossEntropyLoss()
    num_epochs = 3
    print("Starting Greedy Search")
    # Call the greedy search function
    final_model, final_config = greedy_search(
        initial_config, 
        threshold_flops, 
        data_dir, 
        num_classes, 
        num_epochs, 
        batch_size, 
        input_size, 
        mean, 
        std, 
        rank, 
        num_gpus, 
        device_string, 
        criterion, 
        scheduler, 
        dataset_name  # Make sure this variable contains the correct dataset name
    )
    if final_model:
        print(f"Final model configuration: {final_config}")

        final_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{final_config[0]}_depth_{final_config[1]}_width_{final_config[2]}_residual_False-TL.pt"
        torch.save(final_model.state_dict(), final_model_path)


if __name__ == "__main__":
    main()
