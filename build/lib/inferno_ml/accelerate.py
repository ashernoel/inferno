import os
import importlib.util
from .utils import NanoCNN as nanocnn
from .utils.greedy_layer_wise_pruning import greedy_search
import torch.nn as nn
from torchvision import datasets, transforms
from datasets import load_dataset
from .utils import task_specific_LLM_pruning
from .utils import download_model  
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'llama-hf')

def get_default_model_path():
    # Ensure the default model is downloaded if it doesn't exist
    if not os.path.exists(DEFAULT_MODEL_PATH):
        download_model('https://drive.google.com/file/d/1YnO6pqiUhPuAcpIx6Q8t08-6Z9WlbXOq/view?usp=sharing', DEFAULT_MODEL_PATH)
        download_model('https://drive.google.com/file/d/1wxadcqWRi6C0UnUcIZtY9YkdCzKZWZq7/view?usp=sharing', DEFAULT_MODEL_PATH)
    return DEFAULT_MODEL_PATH

def convert_seconds_to_gpu_hours(seconds):
    return seconds / 3600  # Convert training time from seconds to "GPU hours"... simple conversion

def convert_latency_to_flops(max_latency):
    return max_latency * 50e6  # Estimate the compute requirements from latency from graph in Introduction

def check_dataset_directories(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        return False, f"Dataset at {dataset_path} is missing 'train' or 'val' directories."
    return True, None

def get_dataset_path(dataset_name):
    # I might not be able to include these datasets if I want the package to be small enough for PyPi... HM
    predefined_datasets = {
        'imagenet5': '/inferno/datasets/imagenet5',
        'bloodcell': '/inferno/datasets/bloodcell',
        'weather': '/inferno/datasets/weather',
    }
    return predefined_datasets.get(dataset_name, dataset_name)

def load_model_from_path(ckpt_path):
    config_path = os.path.join(ckpt_path, 'config.json')
    config = AutoConfig.from_pretrained(config_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, config=config, padding_side='right')
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, config=config, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    return model, tokenizer

def import_model_from_file(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module

def count_classes_in_dataset(train_dir):
    return len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

def get_mean_std_and_image_size(data_dir, batch_size):
    #statistics for normalization
    transform = transforms.Compose([transforms.ToTensor()])
    image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    total_pixels = 0
    
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        total_pixels += data.size(0) * data.size(2) * data.size(3)  # batch_size * height * width

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    average_image_size = total_pixels / num_batches

    return mean, std, average_image_size

def prune_llama_model(dataset_name, percentage_pruning, with_prompt, ckpt_path, pruned_model_path):
    dataset = load_dataset(dataset_name, split="train")
    task_specific_LLM_pruning.main(ckpt_path=ckpt_path, pruned_model_path=pruned_model_path, percentage=percentage_pruning, dataset=dataset, with_prompt=False, magnitude_based=False, wanda_based=False)
    return pruned_model_path

def accelerate(dataset, max_latency=None, train_time=None, model_type='cnn', model_family=nanocnn.MicroCNN, checkpoint_path=None, pruned_model_path=None, sparsity=0.5):
    # Lots of inputs assumed to be default so suesr does not have to input many

    flops = convert_latency_to_flops(max_latency)
    gpu_hours = convert_seconds_to_gpu_hours(train_time)

    if model_type.lower() == 'cnn':
        # CNN-specific acceleration routine
        dataset_path = get_dataset_path(dataset)
        dataset_check, error_message = check_dataset_directories(dataset_path)
        if not dataset_check:
            return False, error_message
        
        train_dir = os.path.join(dataset_path, 'train')
        num_classes = count_classes_in_dataset(train_dir)
        batch_size = 32
        initial_config = [5, 5, 4]
        num_gpus = 1
        criterion = nn.CrossEntropyLoss()
        mean, std, input_size = get_mean_std_and_image_size(dataset_path, batch_size)
        
        #custom model architecture if specified
        if isinstance(model_family, str) and os.path.isfile(model_family):
            model_module = import_model_from_file(model_family)
            model = model_module.Model()
        else:
            model = model_family()

        final_model, final_config = greedy_search(initial_config, flops, dataset_path, num_classes, train_time, batch_size, input_size, mean, std, rank=0, num_gpus=num_gpus, device_string='cuda:0', criterion=criterion, scheduler=None, dataset_name=dataset)
        return final_model

    elif model_type.lower() == 'llm':

        checkpoint_path = checkpoint_path or get_default_model_path()
        pruned_model_path = pruned_model_path or os.path.join(os.path.dirname(__file__), 'models', 'pruned')
        pruned_model_path = prune_llama_model(dataset, sparsity, checkpoint_path, pruned_model_path)
        model, _ = load_model_from_path(checkpoint_path)
        return model
   
