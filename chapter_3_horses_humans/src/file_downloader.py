# Common file downloader for datasets
import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = os.path.join(dest_folder, url.split('/')[-1])

    # Download the file with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    print(f"--- Downloading: {filename} ---")
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    print("--- Filedownload completed! ---")
    return filename
# Example usage:
# url = 'https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip
# download_file(url, 'data/')
