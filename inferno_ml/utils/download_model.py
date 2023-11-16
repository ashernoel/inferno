import os
import requests
from tqdm import tqdm

def download_model(url, model_dir):
    """
    Download a model file from a given URL into the specified directory.

    Args:
    - url (str): URL where the model is hosted.
    - model_dir (str): Local directory path to save the downloaded model.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(model_dir, local_filename)

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(local_filepath, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    
    print(f"Model downloaded and saved to {local_filepath}.")
    return local_filepath
