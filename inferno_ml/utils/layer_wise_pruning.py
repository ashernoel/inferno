import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import inferno_ml.utils.NanoCNN as se_swish
from inferno_ml.utils.gpu_training_script import get_mean_and_std, train_model, get_data_transforms, optimizer_to, ImageFolderWithPaths
import random
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model_on_gpu(gpu_id, block, depth, width, residual=0):
    with torch.cuda.device(gpu_id):
        net = se_swish.MicroCNN(block, depth, width, residual, 5)
        flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)

        del net
        torch.cuda.empty_cache()

    return flops

def get_data_transforms(input_size, mean, std):
    return {
        'train': transforms.Compose([
			transforms.RandomResizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size+32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

def plot_max_accuracies(ax, results, desired_width, alpha=0.5, linestyle="--"):
    all_accuracies = []
    all_flops = []
    width_accuracies = []
    width_flops = []

    max_accuracy = float('-inf')
    max_flops = 0

    max_accuracy_width = float('-inf')
    max_flops_width = 0

   #  mode = ["imagenet5", "weather", "bloodcell"]
    index = 1

    for row in results:
        accuracy = float(row[4 + index])  # Adjust index based on your CSV format
        flops = int(float(row[3]))  # Adjust index based on your CSV format
        width = int(row[2])  # Adjust index based on your CSV format

        max_accuracy = max(max_accuracy, accuracy)
        max_flops = max(max_flops, flops)

        all_accuracies.append((flops, accuracy))
        if width == desired_width:
            width_accuracies.append((flops, accuracy))

            max_accuracy_width = max(max_accuracy_width, accuracy)
            max_flops_width = max(max_flops_width, flops)

        if flops > 5e9:
            break

    all_accuracies.append((max_flops, max_accuracy))
    width_accuracies.append((max_flops_width, max_accuracy_width))

    # Sort by FLOPs and compute Pareto frontier for all points
    all_accuracies.sort(key=lambda x: x[0])
    max_accuracy = 0
    pareto_accuracies = []
    pareto_flops = []
    for flops, accuracy in all_accuracies:
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            pareto_accuracies.append(accuracy)
            pareto_flops.append(flops)

    pareto_accuracies.append(max_accuracy)
    pareto_flops.append(max_flops)

    # Sort by FLOPs and compute Pareto frontier for points with desired_width
    width_accuracies.sort(key=lambda x: x[0])
    max_accuracy = 0
    pareto_width_accuracies = []
    pareto_width_flops = []
    for flops, accuracy in width_accuracies:
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            pareto_width_accuracies.append(accuracy)
            pareto_width_flops.append(flops)

    pareto_width_accuracies.append(max_accuracy_width)
    pareto_width_flops.append(max_flops_width)

    # Plot the Pareto frontiers
    ax.plot(pareto_flops, pareto_accuracies, alpha=alpha, linestyle=linestyle, color="black", label="Baseline, All Widths")
    ax.plot(pareto_width_flops, pareto_width_accuracies, alpha=alpha, linestyle=linestyle, color="red", label=f"Baseline, Width={desired_width}")


def load_results_from_csv(filename):
    results = []

    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  

        for row in csvreader:
            if not row:
                continue 
            results.append(row)

    return results

def create_model_from_config(config, classes):
    block_count = int(config[0])
    depth = int(config[1])
    width = int(config[2])
    # CHANGE THIS FOR DIFFERENT DATASETS
    num_classes = classes

    model = se_swish.MicroCNN(block_count=block_count, depth=depth, width=width, residual=0, num_classes=num_classes)
    return model


def get_best_configuration(results, desired_width):
    best_config = None
    max_accuracy = -1

    # [imagenet5, weather, bloodcell]
    index = 1

    for row in results:
        width = int(row[2])
        ## UPDATE THIS WITH NEW DATASET 
        accuracy = float(row[4 + index])
        flops = int(float(row[3]))

        if width == desired_width and accuracy > max_accuracy and flops < 5e9:
            max_accuracy = accuracy
            best_config = row

    return best_config

def transfer_weights(old_model, new_model):
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()

    for key in new_state_dict.keys():
        if key.endswith("num_batches_tracked"): 
            continue
        if key in old_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
            new_state_dict[key] = old_state_dict[key]
        elif key in old_state_dict:
            new_state_dict[key] = random_choose_weights(old_state_dict[key], new_state_dict[key])
        else:
            new_state_dict[key].data.uniform_(0, 1)

    new_model.load_state_dict(new_state_dict)
    return new_model

def random_choose_weights(old_weights, new_weights):
    old_shape = old_weights.shape
    new_shape = new_weights.shape
    random_weights = torch.zeros(new_shape)

    for i in range(min(old_shape[0], new_shape[0])):
        if len(old_shape) == 1:
            random_weights[i] = old_weights[i]
        elif len(old_shape) == 2:
            for j in range(min(old_shape[1], new_shape[1])):
                random_weights[i, j] = old_weights[i, j]
        elif len(old_shape) == 3:
            for j in range(min(old_shape[1], new_shape[1])):
                for k in range(min(old_shape[2], new_shape[2])):
                    random_weights[i, j, k] = old_weights[i, j, k]

    return random_weights

def generate_new_config(old_config, new_config):
    new_block_count = min(int(old_config[0]), int(new_config[0]))
    new_depth = min(int(old_config[1]), int(new_config[1]))
    new_width = min(int(old_config[2]), int(new_config[2]))
    return (new_block_count, new_depth, new_width)

def create_graphs(axs, epochs_index, try_width, MODEL_NUM, branch_accuracies, branch_flops, configs_block_decrement, best_config, all_accuracies, num_epochs, results):
    ax2 = axs[(try_width - 1) % 3, epochs_index]  # Assign a subplot to ax2
    plot_max_accuracies(ax2, results, try_width)  # Pass ax2 to your plotting function

    epochs = num_epochs[epochs_index]
    
    if MODEL_NUM == 0:
        ax2.set_title(f"Width: {try_width}, Epochs: {epochs}, One-Model", fontsize=11)
    else: 
        ax2.set_title(f"Width: {try_width}, Epochs: {epochs}, Successive", fontsize=11)
    ax2.set_xlabel("FLOPs", fontsize=9)
    ax2.set_ylabel("Accuracy", fontsize=9)

    for base_index, acc in branch_accuracies.items():
        flops = branch_flops[base_index]
        base_config_block = configs_block_decrement[base_index][0]
        label = f'Block = {base_config_block}'
        ax2.plot(flops, acc, marker='o', linestyle='-', label=label)

    flops = [int(float(config[3])) for config in configs_block_decrement]
    ax2.plot(flops, all_accuracies, marker='o', linestyle='-', label=f'Decrement Blocks, from ({best_config[0]}, {best_config[1]}, {best_config[2]})')
    
    for i in range(len(flops) - 1):
        ax2.annotate("", xy=(flops[i+1], all_accuracies[i+1]), xytext=(flops[i], all_accuracies[i]),
                     arrowprops=dict(arrowstyle="->", lw=1.5),
                     zorder=1)
    
    ax2.legend(loc='best', fontsize='small')
    
    if try_width % 3 == 0 and epochs == num_epochs[-1]:  # Save and display every 9 subplots
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between subplots
        # if MODEL_NUM == 0:
        #     plt.suptitle(f"One-Model: Accuracy vs. FLOPs at Widths [{try_width - 2}, {try_width - 1}, {try_width}] and [{num_epochs[0]}, {num_epochs[0]}, {num_epochs[0]}] epochs ", fontsize=12)
        # else: 
        #     plt.suptitle(f"Successive: Accuracy vs. FLOPs at Widths [{try_width - 2}, {try_width - 1}, {try_width}] and [{num_epochs[0]}, {num_epochs[0]}, {num_epochs[0]}] epochs ", fontsize=12)
        plt.savefig(f"Graph_WEATHER---_Model_{MODEL_NUM}_Widths_to_{try_width}.png")
        plt.show()
        # plt.close('all')  # Close all figures



def main():

    csv_path = "configurations_ALL_flops.csv"
    results = load_results_from_csv(csv_path)

    fig2, axs = plt.subplots(3, 3, figsize=(15, 15))  # Create a grid of 3x3 for subplots

    width_accuracies = {}
    width_flops = {}

    for MODEL_NUM in range(0, 1):  
        for try_width in range (4, 7):
            best_config = get_best_configuration(results, try_width)
            # [imagenet5, weather, bloodcell] ,, index  = 0, 1, or 2 for dataset
            index = 1
            best_config[4] = float(best_config[4 + index])

            ## Update these as well (UTKU)
            # data_dir="/n/idreos_lab/users/usirin/datasets/imagenet_subsets/imagenet_training_5class_subset0_asher"
            # dataset_name="ic_imagenet_5class_subset0"
            data_dir ="/n/idreos_lab/users/usirin/datasets/Multi-class-Weather-Dataset-for-Image-Classification/dataset2"
            dataset_name = "weather"
            # data_dir = "/n/idreos_lab/users/usirin/datasets/blood-cell-images/dataset2-master/dataset2-master/images/TRAIN"
            # dataset_name = "bloodcell"
            num_classes = 0
            train_dir = data_dir + "/train"
            for filename in os.listdir(train_dir):
                num_classes = num_classes + 1

            
            # num_classes = 5
            batch_size=32
            num_epochs=[5, 10, 15]
            rank=0
            num_gpus=1
            input_size = 256

            mean = get_mean_and_std(data_dir, batch_size, input_size)[0]
            std = get_mean_and_std(data_dir, batch_size, input_size)[1]

            for epochs_index, epochs in enumerate(num_epochs):
                    flops_graph = {}
                    accuracies_graph = {}

                    accuracies_graph = {i: [] for i in range(3)}
                    flops_graph = {i: [] for i in range(3)}

                    # Create a list to save branch-wise accuracy and flops for plotting.
                    branch_accuracies = {}
                    branch_flops = {}
 
                    results_accumulated = []  # Accumulate results here
                    # DEPTH DECREMENT 
                    config = best_config
                    old_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{config[0]}_depth_{config[1]}_width_{config[2]}_residual_False.pt"
                    if not os.path.isfile(old_model_path):
                        old_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{config[0]}_depth_{config[1]}_width_{config[2]}_residual_0.pt"

                    old_model = create_model_from_config(config, num_classes)
                    old_model.load_state_dict(torch.load(old_model_path))
                    original_model = old_model

                    # BLOCK DECREMENT   
                    configs_block_decrement = [best_config[:]]
                    configs_block_decrement[0][3] = int(float(configs_block_decrement[0][3]))
                    configs_block_decrement[0][4] = float(configs_block_decrement[0][4])
                    all_accuracies = [float(best_config[4])]

                    for i in range(1, int(best_config[0])):
                        new_config = best_config[:]
                        new_config[0] = int(new_config[0]) - i
                        new_config[3] = evaluate_model_on_gpu(0, int(new_config[0]), int(new_config[1]), int(new_config[2]), 0)
                
                        print(f"old config: {config}, new config: {new_config}")
                        new_model = create_model_from_config(new_config, num_classes)
                        new_model = transfer_weights(old_model, new_model)
                        new_model = new_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

                        data_transforms = get_data_transforms(input_size, mean, std)
                        optimizer = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)
                        scheduler = None
                        last_epoch = 0
                        new_model = new_model.to(rank)
                        device_string = "cuda:" + str(rank)
                        optimizer_to(optimizer,torch.device(device_string))
                        criterion = nn.CrossEntropyLoss()
                        model_name = f"{dataset_name}_block_{int(new_config[0])}_depth_{int(new_config[1])}_width_{int(new_config[2])}_TRANSFERRED"
                        image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
                        image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=num_gpus, rank=rank, shuffle=True, drop_last=False) for x in ['train', 'val']}
                        image_dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2, drop_last=False, sampler=image_samplers[x], persistent_workers=True, prefetch_factor=2) for x in ['train', 'val']}
                        new_model, accuracy = train_model(new_model, num_gpus, data_dir, model_name, rank, image_dataloaders_dict, criterion, optimizer, scheduler, input_size, batch_size, num_epochs=last_epoch+epochs, start_epoch=last_epoch, patience = 10)
                        print(f"ACCURACY AFTER TRANSFER: {accuracy}")
                        if MODEL_NUM == 1:
                            old_model = new_model

                        all_accuracies.append(accuracy)
                        new_config[4] = accuracy
                        configs_block_decrement.append(new_config)

                        print(all_accuracies)

                    print("finished block decrement")
                    print("configs_block_decrement: ", configs_block_decrement)
                    print("all accuracies: ", all_accuracies)
                        

                    for i in range(0, int(best_config[0])):
                        base_config = best_config[:]
                        base_config[0] = int(configs_block_decrement[i][0])
                        base_config[3] = int(float(configs_block_decrement[i][3]))
                        base_config[4] = float(configs_block_decrement[i][4])
                    
                        # Create a list for this branch's accuracy and flops.
                        current_accuracies = [base_config[4]]
                        current_flops = [base_config[3]]

                        if (i > 0 and MODEL_NUM == 1):
                            old_model_path = f"/n/idreos_lab/users/anoel/saved_models/{dataset_name}_block_{int(base_config[0])}_depth_{int(base_config[1])}_width_{int(base_config[2])}_TRANSFERRED.pt"
                            old_model = create_model_from_config(base_config, num_classes)
                            old_model.load_state_dict(torch.load(old_model_path))
                        else: 
                            old_model = original_model
                        

                        print("starting depth decrement for base_config: ", base_config)

                        for depth_decrement in range(1, int(base_config[1])):
                            # Get the new configuration with decremented depth.
                            new_config = list(base_config)
                            new_config[1] = int(new_config[1]) - depth_decrement
                            new_config[3] = evaluate_model_on_gpu(0, int(new_config[0]), int(new_config[1]), int(new_config[2]), 0)
                            print(f"old config: {base_config}, new config: {new_config}")
                            new_model = create_model_from_config(new_config, num_classes)
                            new_model = transfer_weights(old_model, new_model)
                            new_model = new_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

                            data_transforms = get_data_transforms(input_size, mean, std)
                            optimizer = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)
                            scheduler = None
                            last_epoch = 0
                            new_model = new_model.to(rank)
                            device_string = "cuda:" + str(rank)
                            optimizer_to(optimizer,torch.device(device_string))
                            criterion = nn.CrossEntropyLoss()
                            model_name = f"{dataset_name}_block_{int(new_config[0])}_depth_{int(new_config[1])}_width_{int(new_config[2])}_TRANSFERRED"
                            image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
                            image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=num_gpus, rank=rank, shuffle=True, drop_last=False) for x in ['train', 'val']}
                            image_dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2, drop_last=False, sampler=image_samplers[x], persistent_workers=True, prefetch_factor=2) for x in ['train', 'val']}
                            new_model, accuracy = train_model(new_model, num_gpus, data_dir, model_name, rank, image_dataloaders_dict, criterion, optimizer, scheduler, input_size, batch_size, num_epochs=last_epoch+epochs, start_epoch=last_epoch, patience = 10)
                            print(f"ACCURACY AFTER TRANSFER: {accuracy}")

                            if MODEL_NUM == 1:
                                print("ABOUT TO ADD A FEW MORE EPOCHS TO CONFIG (Re-use): ", new_config)
                                old_model = new_model
                                
                            current_accuracies.append(accuracy)
                            current_flops.append(int(float(new_config[3])))

                        # Save the branch's results.
                        branch_accuracies[i] = current_accuracies
                        branch_flops[i] = current_flops

                        print("finished depth decrement for base_config", base_config)
                        print("BRANCH[i] accuracies: ", current_accuracies)
                        print("BRANCH[i] flops: ", current_flops)

                    create_graphs(axs, epochs_index, try_width, MODEL_NUM, branch_accuracies, branch_flops, configs_block_decrement, best_config, all_accuracies, num_epochs, results)
    
                    # # Plot accuracy vs. FLOPs
                    # fig, ax = plt.subplots()
                    # # ax.plot(flops, accuracies, marker='o', linestyle='-')
                    # # Plot for each branch
                    # for base_index, acc in branch_accuracies.items():
                    #     flops = branch_flops[base_index]
                    #     print("index : ", base_index)
                    #     print("flops: ", flops)
                    #     print("acc: ", acc)
                    #     print("configs_block_decrement: ", configs_block_decrement)
                    #     print("configs_block_decrement[index]: ", configs_block_decrement[base_index])
                        
                    #     base_config_block = configs_block_decrement[base_index][0]
                    #     label = f'Block = {base_config_block}'
                    #     ax.plot(flops, acc, marker='o', linestyle='-', label=label)

                    # flops = [int(float(config[3])) for config in configs_block_decrement]
                    # print("flops: ", flops)
                    # print("all_accuracies: ", all_accuracies)
                    # plot_max_accuracies(ax, results, try_width)

                    # ax.plot(flops, all_accuracies, marker='o', linestyle='-', label=f'Decrement Blocks, from ({best_config[0]}, {best_config[1]}, {best_config[2]})')
                    # for i in range(len(flops) - 1):
                    #     ax.annotate("", xy=(flops[i+1], all_accuracies[i+1]), xytext=(flops[i], all_accuracies[i]),
                    #                     arrowprops=dict(arrowstyle="->", lw=1.5),
                    #                     zorder=1)

                    # ax.legend(loc='best')
                    # ax.set_xlabel("FLOPs")
                    # ax.set_ylabel("Accuracy")
                    # if (MODEL_NUM == 0):
                    #     ax.set_title(f"Accuracy vs. FLOPs: Width = {try_width}, One-Model, {epochs} epochs")
                    # else:
                    #     ax.set_title(f"Accuracy vs. FLOPs: Width = {try_width}, Successive, {epochs} epochs")

                    # plt.show()
                    # plt.savefig(f"flops_vs_accuracy_transfer_learning_{dataset_name}_width={try_width}-{MODEL_NUM}-{epochs} epochs.png", dpi=300)

    #                 width_accuracies[try_width] = all_accuracies
    #                 accuracies_graph[MODEL_NUM].append(all_accuracies)
    #                 flops_graph[MODEL_NUM].append(flops)
    #                 width_flops[try_width] = flops 

    # # Saving or displaying the figures
    # for i, fig in enumerate(figures):
    #     fig.suptitle(f'Figure {i+1}')
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.95)
    #     fig.savefig(f'figure_{i+1}.png')
    #     plt.close(fig)

if __name__ == "__main__":
    main()



