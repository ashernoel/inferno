import torch
import torch.nn.utils.prune as prune
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoConfig,
    Trainer
)
from datasets import load_dataset
import random
import numpy as np
import torch.optim as optim
import time
import argparse
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image


def get_pruning_path(percentage):
    """Get pruning path string based on the percentage."""
    percentage_mapping = {
        0.1: "ten_percent_pruning",
        0.2: "twenty_percent_pruning",
        0.3: "thirty_percent_pruning",
        0.4: "forty_percent_pruning",
        0.5: "fifty_percent_pruning",
        0.6: "sixty_percent_pruning",
        0.7: "seventy_percent_pruning",
        0.8: "eighty_percent_pruning",
        0.9: "ninety_percent_pruning",
        # Add more mappings if necessary
    }
    return percentage_mapping.get(percentage, "unknown_percentage")

def magnitude_pruning(model, percentage):
    def prune_module(module):
        # Get absolute values of weights
        abs_weights = torch.abs(module.weight.data)

        # Calculate the threshold
        num_elements_to_prune = int(abs_weights.numel() * percentage)
        threshold, _ = torch.kthvalue(abs_weights.view(-1), num_elements_to_prune)

        # Create a mask and apply pruning
        mask = abs_weights >= threshold
        module.weight.data.mul_(mask.float())

    for name, module in model.named_modules():
        # Prune weights in Linear layers within MLP or Self Attention blocks
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):
            prune_module(module)


sample_number = 0
num_new_tokens = 50

def visualize_results(tensor, module_number, sample_number):
    plt.figure(figsize=(10, 10))
    tensor_cpu = tensor.cpu()
    
    # Convert the tensor to a numpy array
    matrix = tensor_cpu.numpy()
    
    # Calculate the 1st and 99th percentiles of the data to exclude extreme values
    vmin, vmax = np.percentile(matrix, [1, 99])
    
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    global num_new_tokens
    input_number = sample_number // num_new_tokens + 1
    # plt.colorbar(label='Cumulative Absolute Value for Input #{input_number}'.format(input_number=input_number))
    plt.colorbar(label='Cumulative Absolute Value for Input #1; Approach: No First Token')

    
    # Mapping module numbers to layer names
    module_mapping = {
        0: "Q layer",
        1: "K layer",
        2: "V layer",
        3: "O layer",
        4: "1st Feed Forward Network",
        5: "2nd Feed Forward Network",
        6: "3rd Feed Forward Network",
    }

    if sample_number == -1: 
        title = f"Cumulative Activations of {module_mapping.get(module_number, module_number)}, Original Weights, Block: 1 of 32"
    else:
        title = f"Cumulative Activations of {module_mapping.get(module_number, module_number)}, Token: {sample_number}, Block: 1 of 32"
    plt.title(title)
    
    plt.xlabel('Dimension')
    plt.ylabel('Dimension')
    
    # Save the figure to disk
    file_path = os.path.join("images_no-first-token_prompt2", f"{module_number}__{sample_number:02d}.png")
    plt.savefig(file_path)
    plt.close()  # Close

def create_gif(module_number):
    images = []
    filenames = sorted(os.listdir("images_no-first-token_prompt2"))
    
    print("got heereeee")

    for filename in filenames:
        if filename.startswith(f"{module_number}__"):
            images.append(imageio.imread(os.path.join("images_no-first-token_prompt2", filename)))
             # Debugging: Printing image shapes
            img_path = os.path.join("images_no-first-token_prompt2", filename)
            img = Image.open(img_path)
            
            print(f"Image {filename} shape:", img.size)
    
    print("here")
    for i in [3, 6, 9, 12]:
        gif_path = os.path.join("images_no-first-token_prompt2", f"{module_number}_{i}.gif")
        print("here2")
        imageio.mimsave(gif_path, images, fps=i)


def update_skews(skews, percentage, quantile_cutoff):
    batch_size = 32  # you can adjust the batch_size based on your GPU memory
    num_blocks, num_rows, _ = skews.shape

    dataset_sizes = torch.tensor([4096] * 6 + [0], device='cuda')
    percentage = torch.tensor(percentage).cuda()

    skews = skews.view(-1, 7)  # reshaping to 2D tensor for easier batching
    num_batches = (num_blocks * num_rows) // batch_size

    for batch_idx in range(0, num_batches * batch_size, batch_size):
        skew_batch = skews[batch_idx:batch_idx+batch_size]

        new_values = quantile_cutoff(skew_batch, dataset_sizes, percentage)

        new_values[:, 6] = percentage # set 7th skew = to percentage, since it's a hassle to updat

        skews[batch_idx:batch_idx+batch_size] = new_values  # update the skews tensor with new values

    skews = skews.view(num_blocks, num_rows, 7)  # reshaping back to original shape
    return skews


def quantile_cutoff(skews, dataset_sizes, sparsity):
    total_items = dataset_sizes.sum()
    alpha = torch.zeros((32, 1), dtype=torch.float32, device='cuda')  # Modified this line

    iter_count = 0

    while True:
        quantile_cutoffs = torch.zeros_like(skews, device='cuda')  # Initializing the tensor
        
        # Iterating over each row
        for i in range(skews.shape[0]):
            quantile_cutoffs[i] = torch.clamp(1 - sparsity + (skews[i] - sparsity)/2 - alpha[i], 0.05, 1)
        
        weighted_quantiles_sum = torch.sum(quantile_cutoffs[:, :6], dim=1, keepdim=True)
        distance = (weighted_quantiles_sum / 6) - sparsity
        
        alpha += distance 
        iter_count += 1

        if abs(distance).mean() <= 0.01 or iter_count > 1:
            break

    return quantile_cutoffs



def custom_skew(input_tensor):
    # Ensure the input tensor does not contain NaN or inf values
    if torch.any(torch.isnan(input_tensor)) or torch.any(torch.isinf(input_tensor)):
        print("ERRNaN or inf found in input tensor")
        return torch.tensor(float('nan'))  # or handle it as you see fit
    
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor, unbiased=False) + 1e-8  # Adding epsilon to avoid division by zero
    
    # Check if standard deviation is very close to zero (considering epsilon)
    if std < 1e-8:
        print("RR: Standard deviation is very close to zero")
        return torch.tensor(float('nan'))  # or handle it as you see fit
    
    adj_factor = ((input_tensor.numel()) ** 0.5) / (input_tensor.numel() - 1)
    skewness = adj_factor * torch.mean((input_tensor - mean) ** 3) / (std ** 3)
    
    return skewness

def calculate_kurtosis(tensor):
    mean = torch.mean(tensor)
    std_dev = torch.std(tensor)
    n = tensor.numel()
    
    if std_dev.item() == 0:
        # If the standard deviation is zero, returning zero or a message indicating no skewness
        return 0
    else:
        skewness = (1/n) * torch.sum(((tensor - mean) / std_dev) ** 3)
        return skewness

def pearson_skewness(input_tensor, percentage):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically choose the device
    
    # Transfer tensor to the appropriate device
    input_tensor = input_tensor.to(device)
    
    mean = torch.mean(input_tensor)
    median = torch.median(input_tensor)
    std = torch.std(input_tensor, unbiased=False) + 1e-8  # Adding epsilon to avoid division by zero

    skewness = 3 * (mean - median) / std

    # if skew is not a number of inf, return 0.5
    if torch.isnan(skewness).item() or torch.isinf(skewness).item():
        return percentage
    
    return skewness

    ## lower means more/less skewed 


def batch_pearson_skewness(matrix, percentage):
    mean = torch.mean(matrix, dim=1)
    median, _ = torch.median(matrix, dim=1)
    std = torch.std(matrix, dim=1, unbiased=False) + 1e-8

    skewness = 3 * (mean - median) / std
    skewness[torch.isnan(skewness) | torch.isinf(skewness)] = 1 - percentage # if 0.3, then 0.7 should be returned, for example
    
    ## if mean and median are 0, then skewness is 0, so 0.5 is returned

    return 1 - skewness

def product_pruning(model, percentage, dataset, tokenizer, with_prompt):

    total_data_points = len(dataset)
    print("Total data points: ", total_data_points)
    # n = 1000
    n = 100
    print("Size of calibration set: ", total_data_points // n)
    global sample_size
    global num_new_tokenss  
    sample_size = total_data_points // n 

    activations_start = torch.zeros((14, num_new_tokens*sample_size), device='cuda:0')
    activations_start = torch.zeros((14, num_new_tokens*sample_size), device='cuda:0')


    def hook_fn(module, input, output):
        global sample_number


        module = module.to('cuda:0')

        input_modified = input[0][0].to('cuda:0')  
        # print("input size: ", input_modified.size())
        l2_norms = torch.norm(input_modified, p=2, dim=0, keepdim=True)
        l2_norms = l2_norms.to('cuda:0')
        epsilon = 1e-10
        result = torch.abs(module.weight) * (l2_norms + epsilon)

        # if (torch.abs(module.weight) == float('inf')).any():
        #     print("Module weights contain infs.")
        
        # if (l2_norms == float('inf')).any():
        #     print("L2 norms contain zeros or infs.")

                   # The rest of your code remains unchanged
            ## only save the later tokens 
        if (module in product_values):
            product_values[module] += result
        elif ((sample_number // 224 ) > 0):
            product_values[module] = result


        # print sum of activations 
        # if module in product_values:
        #     if sample_number % 224 < 7: 
        #         actual_sample_number = sample_number // 224
        #         print("actual token number: ", actual_sample_number)
        #         print("sample number ", actual_sample_number // 50)
        #         print("module number in 1st block: ", sample_number % 224)
        

        #         print("sum of activations: ", torch.sum(result).item())
        #         activations_start[sample_number % 224][sample_number // 224] = torch.sum(result).item()
        #         # module_number = sample_number % 224 
        #         # if (actual_sample_number == 0):
        #         #     visualize_results(module.weight, module_number, -1) # show the original weight values
            
        #         # visualize_results(product_values[module], module_number,actual_sample_number)
        #         # if (actual_sample_number == num_new_tokens - 1):
        #         #     print("creating GIF")
        #         #     create_gif(module_number)
        #         #     print("finished")
        #     if sample_number % 224 > 216:
        #         print("  module number in last block: ", sample_number % 224)
        #         print("  Any NaNs: ", torch.isnan(result).any())
        #         print("  Any Infs: ", (result == float('inf')).any() | (result == float('-inf')).any())
        #         clipped_result = torch.clamp(result/1000, min=-1e10, max=1e10)
        #         print("  sum of activations: ", torch.sum(clipped_result).item())
        #         activations_start[sample_number % 224 - 210][sample_number // 224] = torch.sum(clipped_result).item()


        # if module in product_values:
        #     if sample_number % 224 < 7: 
        #         actual_sample_number = sample_number // 224
        #         module_number = sample_number % 224 
        #         if (actual_sample_number == 0):
        #             visualize_results(module.weight, module_number, -1) # show the original weight values
            
        #         visualize_results(product_values[module], module_number,actual_sample_number)
        #         if (actual_sample_number == num_new_tokens - 1):
        #             print("creating GIF")
        #             create_gif(module_number)
        #             print("finished")
        
        sample_number += 1

        torch.cuda.empty_cache()


    product_values = {}
    hooks = []
    for name, module in model.named_modules():
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Randomly select 10% of the dataset
    random_indices = random.sample(range(total_data_points), sample_size)
    subset = [dataset[i] for i in random_indices]

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Setting padding side to left


    for batch in subset:
        print(batch)
        # print("BATCH", batch)
        # text_data = batch['passage'] + ' ' + batch['question'] ## FOR BOOLQ
        arabic_text_data = batch['translation']['ar']
        english_text_data = batch['translation']['en']

        # Randomly choose between Arabic and English with a 50% probability
        text_data = arabic_text_data if random.random() < 0.5 else english_text_data

        # text_data = batch['ctx_a'] + ' ' + batch['ctx_b']
        # if with_prompt:
        #     text_data = batch['ctx_a'] + ' Thinking step-by-step, ' + batch['ctx_b']
        
        inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        print("INPUTS TEXT: ", text_data)
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}  # Moving everything to cuda:0
        attention_mask = inputs['attention_mask']  # The tokenizer now automatically creates an attention mask
        
        with torch.no_grad():
            # outputs = model(**inputs)
            generate_ids = model.generate(input_ids=inputs['input_ids'], 
                                      attention_mask=attention_mask, 
                                      max_new_tokens=num_new_tokens,  # Adjust this based on how long you expect the outputs to be
                                      pad_token_id=tokenizer.eos_token_id,  # Setting pad_token_id
                                      do_sample=True,
                                      use_cache=True,  # You might want to enable or disable this based on your needs
                                      )
            decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print("GENERATED TOKENS: ", generate_ids[0].shape)
    
        # Convert to numpy array
    ac_start = activations_start.cpu().numpy()


    # Set numpy print options
    np.set_printoptions(threshold=np.inf)

# Function to clean data
    def clean_data(data_row):
        # Exclude the first data point
        data_row = data_row[1:]
        # data_row = [x if x != 0.00 else np.nan for x in data_row]
        data_row = [13 if np.isinf(x) else x for x in data_row]

        mean = np.mean(data_row)
        std_dev = np.std(data_row)
        cleaned_data = [x if (mean - 4*std_dev) <= x <= (mean + 4*std_dev) else np.nan for x in data_row]
        return cleaned_data

    # Function to create and save plots
    def create_save_plot(data_row, token_index):
        cleaned_data_row = clean_data(data_row)  # Clean the data
        plt.figure(figsize=(10, 5))
        plt.plot(cleaned_data_row)
        plt.title(f'Layer {token_index} of 224: Token Activations')
        plt.xlabel('Tokens')
        plt.ylabel('Activation')
        plt.savefig(f'token_activations/token_{token_index}.png')
        plt.close()

    # Create and save a plot for each token
    # Assuming ac_start is your data
    for i in range(ac_start.shape[0]):
        create_save_plot(ac_start[i], i)

    # Print the full array
    print("ACIVATIONS START: ", ac_start)
    # Remove hooks
    for hook in hooks:
        hook.remove()

    ## CALCLULATE SKEWS
    skews = torch.zeros((32, 11008, 7), device='cuda:0') # Initializing the skew tensor
    # add percentage to entire tensor
    skews[:, :, :] = percentage

    layer_counter = 0  # Counter to keep track of each layer
    block_counter = 0  # Counter to keep track of each block of 7 layers
    for name, module in model.named_modules():
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):

            prod_values = product_values[module].to('cuda')  # Make sure product_values are on CUDA
            abs_prod_values = torch.abs(prod_values)
            
            # print block number and module name: 
            # print("Block: ", block_counter)
            # print("Module name: ", name)
            # Calculating the skewness for each row
            # Calculate skewness for each row in batch
            skews_per_row = batch_pearson_skewness(abs_prod_values, percentage)
                
            # Check the length of skews_per_row
            length_to_pad = 11008 - skews_per_row.size(0)

            if length_to_pad > 0:
                padding = torch.zeros(length_to_pad, device='cuda')  # Create padding on the GPU
                skews_per_row = torch.cat([skews_per_row, padding])
            
            skews[block_counter, :, layer_counter] = skews_per_row  # Move tensor back to CPU if needed

            layer_counter += 1  # Move to the next layer

            # If 7 layers are processed, move to the next block
            if layer_counter % 7 == 0:
                layer_counter = 0  # Reset layer counter
                block_counter += 1  # Move to the next block
                print("Block: ", block_counter)
    
    print("CHECK1: calculated all skews")
    print(skews[0, :10, :])
    
    skews = skews.cuda()
        # UPDATE SKEWS
    print("starting loop")
    updated_skews = update_skews(skews, percentage, quantile_cutoff)

    print("CHECK2: Updated Skews")

    # print the first 10 rows of skews
    print(skews[0, :10, :])

    layer_counter = 0  # Counter to keep track of each layer
    block_counter = 0  #
    for name, module in model.named_modules():
        
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):
            
            prod_values = product_values[module].to('cuda')  # Make sure product_values are on CUDA

            for i in range(abs_prod_values.size(0)):  # Loop over the rows
                
                row_prod_values = prod_values[i].float() 
                # Retrieve the quantile from the updated skews tensor
                quantile = skews[block_counter, i, layer_counter].item()
                
                # TODO: find the number in row_prod_values that corresponds to the quantile-percentile
                # if 0.90, then it will give number that 90% of numbers are LESS than 
                threshold = torch.quantile(row_prod_values, quantile)

                # TODO: Create a mask by comparing each element to the respective threshold, 1 if larger 
                mask = row_prod_values > threshold

                # Ensure that the mask is on the same device as the module weights
                mask = mask.to(module.weight.device)
                module.weight.data[i].mul_(mask.float()) 
                torch.cuda.empty_cache()  # ty_cache()  # Release unused memory
                                    # ## ROW BASED
            # print("GOT HEREEE")
            # prod_values = product_values[module]
            # # Get the absolute values of the product values
            # # visualize_matrix(prod_values.cpu().numpy(), name, "before")
            # abs_prod_values = torch.abs(prod_values)
            
            # # Initialize a tensor to store the thresholds for each row
            # thresholds = torch.zeros(abs_prod_values.size(0), device=abs_prod_values.device)
            
            # # Calculate the threshold for each row
            # for i in range(abs_prod_values.size(0)):
            #     row_values = abs_prod_values[i].cpu().numpy()  # Move tensor to CPU and convert to numpy for quantile calculation
            #     thresholds[i] = np.quantile(row_values, percentage)  # Calculate the threshold based on the absolute values
            
            # # Extend the threshold to have the same shape as abs_prod_values for element-wise comparison
            # thresholds = thresholds.unsqueeze(1).expand_as(abs_prod_values)
            
            # # Create the mask by comparing each element to the respective threshold
            # mask = abs_prod_values >= thresholds
            
            # # Apply the mask to the module's weights
            # module.weight.data.mul_(mask.float())

            layer_counter += 1  # Move to the next layer

             # If 7 layers are processed, move to the next block
            if layer_counter % 7 == 0:
                layer_counter = 0  # Reset layer counter
                print(block_counter)
                block_counter += 1  # Move to the next block
    
def l2_pruning(model, percentage, dataset, tokenizer, with_prompt):

    def hook_fn(module, input, output):
        global sample_number

        module = module.to('cuda:0')

        input_modified = input[0][0].to('cuda:0')  

        ## L2 NORM 
        l2_norms = torch.norm(input_modified, p=2, dim=0, keepdim=True)
        l2_norms = l2_norms.to('cuda:0')
        result = torch.abs(module.weight) * l2_norms

                   # The rest of your code remains unchanged
        if module in product_values:
            product_values[module] += result
            ## only save the later tokens 
        elif (sample_number // 224 > 0):
            product_values[module] = result

        
        sample_number += 1
        torch.cuda.empty_cache()


    product_values = {}
    hooks = []
    for name, module in model.named_modules():
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    total_data_points = len(dataset)
    print("Total data points: ", total_data_points)
    # n = 1000a
    n = 100
    print("Size of calibration set: ", total_data_points // n)
    global sample_size
    sample_size = total_data_points // n  # This will give you 80 datapoints, 40k training 
    # Randomly select 10% of the dataset
    random_indices = random.sample(range(total_data_points), sample_size)
    subset = [dataset[i] for i in random_indices]

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Setting padding side to left
    print("GOT HERE")
    for batch in subset:
        print(batch)
        # text_data = batch[    'sentence1'] + ' ' + batch['sentence2'] # RTE

        arabic_text_data = batch['translation']['ar']
        english_text_data = batch['translation']['en']

        # Randomly choose between Arabic and English with a 50% probability
        text_data = arabic_text_data if random.random() < 0.5 else english_text_data
        # text_data = batch['passage'] + ' ' + batch['question'] # BoolQ
        # print("BATCH: ", batch)
        # text_data = batch['ctx_a'] + ' ' + batch['ctx_b']
        # if with_prompt:
        #     text_data = batch['ctx_a'] + ' ' + batch['ctx_b']
        
        inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}  # Moving everything to cuda:0
        attention_mask = inputs['attention_mask']  # The tokenizer now automatically creates an attention mask
        # print("Here")
        
        with torch.no_grad():
            # outputs = model(**inputs)
            global num_new_tokens
            generate_ids = model.generate(input_ids=inputs['input_ids'], 
                                      attention_mask=attention_mask, 
                                      max_new_tokens=num_new_tokens,  # Adjust this based on how long you expect the outputs to be
                                      pad_token_id=tokenizer.eos_token_id,  # Setting pad_token_id
                                      do_sample=True,
                                    #   use_cache=False  # You might want to enable or disable this based on your needs
                                      )
            decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print("GENERATED TOKENS: ", generate_ids[0].shape)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    for name, module in model.named_modules():
        # if "mlp" in name.lower() and isinstance(module, torch.nn.Linear):
        if ("mlp" in name.lower() or "self_attn" in name.lower()) and isinstance(module, torch.nn.Linear):
                        
            prod_values = product_values[module]
            # Get the absolute values of the product values
            # visualize_matrix(prod_values.cpu().numpy(), name, "before")
            abs_prod_values = torch.abs(prod_values)
            
            # Initialize a tensor to store the thresholds for each row
            thresholds = torch.zeros(abs_prod_values.size(0), device=abs_prod_values.device)
            
            # Calculate the threshold for each row
            for i in range(abs_prod_values.size(0)):
                row_values = abs_prod_values[i].cpu().numpy()  # Move tensor to CPU and convert to numpy for quantile calculation
                thresholds[i] = np.quantile(row_values, percentage)  # Calculate the threshold based on the absolute values
            
            # Extend the threshold to have the same shape as abs_prod_values for element-wise comparison
            thresholds = thresholds.unsqueeze(1).expand_as(abs_prod_values)
            
            # Create the mask by comparing each element to the respective threshold
            mask = abs_prod_values >= thresholds
            
            # Apply the mask to the module's weights
            module.weight.data.mul_(mask.float())


def main(ckpt_path, pruned_model_path, percentage, dataset, with_prompt=False, magnitude_based=False, wanda_based=False):
    try: 
        print(f"Loading model from {ckpt_path}...")
        
        # Load config and model
        # Check if config.json exists in the folder, if not you might need to load it differently
        if 'config.json' in os.listdir(ckpt_path):
            config = AutoConfig.from_pretrained(ckpt_path)
            print(config)
        else:
            # Specify how you want to load the config if it's not in the model directory
            config = AutoConfig.from_pretrained('path/to/your/config.json') 

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, config=config, padding_side='right')
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            config=config, 
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.to('cuda:0')  # or 'cuda:1' depending on which GPU you want to use



        print("Model loaded successfully")

        # Apply pruning
        if magnitude_based:
            magnitude_pruning(model, percentage)
        elif traditional: 
            l2_pruning(model, percentage, dataset, tokenizer, with_prompt)
        else: 
            product_pruning(model, percentage, dataset, tokenizer, with_prompt)


        # if (LoRA_finetune == True):
        #     print("Beginning LoRA finetuning...")
        #     lora_finetuning(model, dataset, tokenizer, pruned_model_path)
        # else: 
        #     # Save pruned model
        print(f"Saving pruned model to {pruned_model_path}...")
        model.save_pretrained(pruned_model_path)
        print(f"Pruned model saved at {pruned_model_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage", type=float, required=False, default=None, help="Pruning percentage.")
    parser.add_argument("--mode", type=str, required=False, default=None, help="Model mode: with or without prompt.")
    
    args = parser.parse_args()

    # Set default values if arguments are None or empty strings
    if args.percentage is None or args.percentage == "":
        args.percentage = 0.5  # replace YOUR_DEFAULT_PERCENTAGE with your default value
    if args.mode is None or args.mode.strip() == "":
        args.mode = "traditional/test15arabic"  # replace YOUR_DEFAULT_MODE with your default value
    
    pruning_path = get_pruning_path(args.percentage)
    base_path = f"/n/idreos_lab/users/anoel/pruned_llama/product/{pruning_path}/"
    pruned_model_path = f"{base_path}{args.mode}"
    
    with_prompt = 'with_prompt' in args.mode
    magnitude_based = 'magnitude_based' in args.mode
    traditional = 'traditional' in args.mode
    print("entering main loop soon")
    
    dataset = load_dataset("iwslt2017", "iwslt2017-ar-en", split="test")
    main("/n/idreos_lab/users/anoel/llama/llama-hf", pruned_model_path, args.percentage, dataset, with_prompt, magnitude_based, traditional)
