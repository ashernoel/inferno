from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import sys
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import MultiStepLR
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
import inferno_ml.utils.NanoCNN as se_swish
from inferno_ml.utils.flop_list_generator import generate_flop_list
import csv
from torch.cuda import FloatTensor
import gc
from collections import OrderedDict
import torch.distributed as dist





class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index): 
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # this number might create a problem. if you submit two jobs and they are scheduled on the same machine, you cannot run two jobs on the same port number on the same machine. you can modify it to make it dynamically allocated. any number starting and increasing from 12355 is fine: 12355, 12356, 12357, ...
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def save_model_config_and_accuracy_to_csv(flop_list, accuracies, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["block_count", "depth", "width", "residual", "flops", "cluster", "accuracy"])

        for i in range(len(flop_list)):
            model_config = flop_list[i]
            accuracy = accuracies[i]
            csv_writer.writerow(model_config + (accuracy,))


def save_model_config_and_accuracy_to_csv_target_flops(flop_list, accuracies, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["cluster", "block_count", "depth", "width", "residual", "flops", "accuracy"])

        for i in range(len(flop_list)):
            model_config = flop_list[i]
            accuracy = accuracies[i]
            csv_writer.writerow(model_config + (accuracy,))



# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def load_results_from_csv(filename):
    flop_list = []

    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row

        for row in csvreader:
            b, d, w, r, formatted_flops, cluster = row
            flops = float(formatted_flops.replace('inf', '1e9999'))
            flop_list.append((int(b), int(d), int(w), r == 'True', flops, int(cluster)))
    return flop_list

def check_val_acc(model, num_gpus, dataset_name, rank, dataloader, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    phase = 'val'
    best_acc = 0.0

    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    num_processed_items = 0
    for inputs, labels, _ in dataloader:
        # Print everything relevant:
        inputs = inputs.to(rank)
        labels = labels.to(rank)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
         
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0) # inputs.size(0) is batch size
        running_corrects += torch.sum(preds == labels.data)
        num_processed_items += inputs.size(0)

    epoch_loss = running_loss / num_processed_items 
    epoch_loss_tensor = torch.Tensor([epoch_loss]).to(rank)
    # sync_epoch_loss = reduce_tensor(epoch_loss_tensor, num_gpus)
    # sync_epoch_loss_item = to_python_float(sync_epoch_loss)
    sync_epoch_loss_item = to_python_float(epoch_loss_tensor)

    epoch_acc = running_corrects.double() / num_processed_items 
    epoch_acc_tensor = torch.Tensor([epoch_acc]).to(rank)
    # sync_epoch_acc = reduce_tensor(epoch_acc_tensor, num_gpus)
    # sync_epoch_acc_item = to_python_float(sync_epoch_acc)
    sync_epoch_acc_item = to_python_float(epoch_acc_tensor)

    if rank == 0:
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, sync_epoch_loss_item, sync_epoch_acc_item))
        print()

    val_acc_history.append(sync_epoch_acc_item)
 
    time_elapsed = time.time() - since
    if rank == 0:
        print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)

    return model, val_acc_history

def rgb2ycrcb_torch_v2(images,ycrcb):
    ycrcb[:,0,:,:] = images[:,0,:,:] * .299   + images[:,1,:,:] * .587   + images[:,2,:,:] * .114
    ycrcb[:,1,:,:] = images[:,0,:,:] * .5     + images[:,1,:,:] * -.4187 + images[:,2,:,:] * -.0813
    ycrcb[:,2,:,:] = images[:,0,:,:] * -.1687 + images[:,1,:,:] * -.3313 + images[:,2,:,:] * .5
    ycrcb[:,[1,2],:,:] += 0.5

    return ycrcb

def train_model(model, num_gpus, data_dir, model_name, rank, dataloaders, criterion, optimizer, scheduler, input_size, batch_size, num_epochs, start_epoch, patience=10):

    since = time.time()

    best_acc = 0.0
    epochs_no_improve = 0

    # path variables
    train = "train"
    val = "val"

    val_accuracies = []
    val_accuracies_graph = []
    epochs = []
    epochs_graph = []

    if start_epoch >= num_epochs and rank == 0:
        print("starting epoch is less than or equal to number of epochs. exiting...")
        return model

    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train, val]:
            if phase == train:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Set epoch for distributed sampler
            dataloaders[phase].sampler.set_epoch(epoch)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            num_processed_items = 0

            # accumulation_steps = 16  # Adjust this value as needed

            for i, (inputs, labels, _) in enumerate(dataloaders[phase]):
                torch.cuda.empty_cache()
                # move the batch to gpu
                inputs = inputs.to(rank)
                labels = labels.to(rank)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward
                with torch.set_grad_enabled(phase == train):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == train:
                        # loss = loss / accumulation_steps  # Normalize loss
                        loss.backward()
                        
                        # if (i+1) % accumulation_steps == 0:  # Update weights every accumulation_steps
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                if phase == val:
                    # statistics only for val
                    running_loss += loss.item() * inputs.size(0)  # inputs.size(0) is batch size
                    running_corrects += torch.sum(preds == labels.data)
                    num_processed_items += inputs.size(0)

            if phase == val:
                epoch_loss = running_loss / num_processed_items
                epoch_loss_tensor = torch.Tensor([epoch_loss]).to(rank)
                sync_epoch_loss_item = to_python_float(epoch_loss_tensor)

                epoch_acc = running_corrects.double() / num_processed_items
                epoch_acc_tensor = torch.Tensor([epoch_acc]).to(rank)
                sync_epoch_acc_item = to_python_float(epoch_acc_tensor)

                if rank == 0:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, sync_epoch_loss_item, sync_epoch_acc_item))

                # Update the list of validation accuracies and epochs
                val_accuracies.append(sync_epoch_acc_item)
                val_accuracies_graph.append(sync_epoch_acc_item)
                epochs_graph.append(epoch)
                epochs.append(epoch)

                if len(val_accuracies) > 4:
                    val_accuracies.pop(0)
                    epochs.pop(0)

                # deep copy the model if it's better than the best
                if sync_epoch_acc_item > best_acc:
                    best_acc = sync_epoch_acc_item
                    model_path = f"/n/idreos_lab/users/anoel/saved_models/best_{model_name}.pt"
                    torch.save(model.state_dict(), model_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            # take a scheduler step to reduce learning rate
        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            print()

    time_elapsed = time.time() - since
    if rank == 0:
        # Save the trained model
        model_path = f"/n/idreos_lab/users/anoel/saved_models/last_{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        
        print("SAVED MODEL!! at path ", model_path)
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # Calculate the average of the ten most recent validation accuracies
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0.0

    return model, avg_val_accuracy


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

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

def main(local_rank, num_gpus, data_dir, dataset_name, num_classes, batch_size, input_size, num_epochs, mean, std, model_config):

    # Must do this before any broadcasting
    setup(local_rank, num_gpus)

    model = se_swish.MicroCNN(int(model_config[0]), int(model_config[1]), int(model_config[2]), int(model_config[3]), num_classes)
    model = model.to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    last_epoch = 0
    device_string = "cuda:" + str(local_rank)
    optimizer_to(optimizer,torch.device(device_string))
        

    criterion = nn.CrossEntropyLoss()

    data_transforms = get_data_transforms(input_size,mean,std)

    model_name = f"{dataset_name}_block_{model_config[0]}_depth_{model_config[1]}_width_{model_config[2]}_residual_{model_config[3]}"

    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    input_example = image_datasets['train'][0][0]
    input_example = input_example.unsqueeze(0).to(local_rank)  # Add batch dimension and send to device
    output_example = model(input_example)

    sub_directory = "model_graphs"
    os.makedirs(sub_directory, exist_ok=True)

    image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=num_gpus, rank=local_rank, shuffle=True, drop_last=False) for x in ['train', 'val']}
    image_dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=8, drop_last=False, sampler=image_samplers[x], persistent_workers=True, prefetch_factor=2) for x in ['train', 'val']}
    model, val_acc_history = train_model(model, num_gpus, data_dir, model_name, local_rank, image_dataloaders_dict, criterion, optimizer, None, input_size, batch_size, num_epochs=last_epoch+num_epochs, start_epoch=last_epoch, patience = 10)
 

    print(val_acc_history)

    print("Finished training model: ", model_name)

    del model
    gc.collect()

    cleanup()
    print("val_acc_history: ", val_acc_history)
    return val_acc_history


def get_mean_and_std(data_dir, batch_size, input_size):
    print("Calculating mean & std for normalization...")
    # create dataset & data loader
    transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.ToTensor()])
    image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print("mean & std:", mean, std)

    return mean, std

if __name__ == '__main__':

    cwd = os.getcwd()
    os.makedirs(cwd + "/saved_models", exist_ok=True)

    # # parse argument and save to id
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--id', type=int, default=0)
    # args = parser.parse_args()
    # id = args.id

    data_dir ="/n/idreos_lab/users/usirin/datasets/full_imagenet"
    dataset_name = "imagenet1000"


    num_classes = 0
    train_dir = data_dir + "/train"
    for filename in os.listdir(train_dir):
        num_classes = num_classes + 1
    input_size = 256
    batch_size = 128
    # mean = get_mean_and_std(data_dir, batch_size, input_size)[0]
    # std = get_mean_and_std(data_dir, batch_size, input_size)[1]

    num_gpus = 4
    num_epochs = 200

    mean = [0.4811, 0.4575, 0.4079] # from the output 
    std = [0.2766, 0.2693, 0.2833]
    print("testttt")
    # mean = [5.071e-01, 4.585e-01, 3.928e-01]
    # std = [2.197e-01, 2.173e-01, 2.153e-01]

    model_config = [5, 5, 5, 0] # default model config. 1 block, 1 depth, 1 width

    mp.spawn(
        main,
        args=(num_gpus,data_dir,dataset_name,num_classes, batch_size, input_size, num_epochs, mean, std, model_config),
        nprocs=num_gpus
    )