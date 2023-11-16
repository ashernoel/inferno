#!/bin/bash

dataset_name="ic_imagenet_5class_subset0"
data_dir="/n/idreos_lab/users/usirin/datasets/imagenet_subsets/imagenet_training_5class_subset0_asher"
input_size=256
num_gpus=2
batch_size=64 # per-gpu
num_epochs=1

# comment out to write results to stdout
python gpu_training_script.py $data_dir $dataset_name $input_size $num_gpus $num_epochs $batch_size
# comment out to write results to a file
#python multi_gpu_training.py $data_dir $dataset_name $input_size $num_gpus $num_epochs $batch_size > out_$dataset_name\_$num_epochs 2>&1

