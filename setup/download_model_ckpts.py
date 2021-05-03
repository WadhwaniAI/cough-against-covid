"""
Downloads pre-trained model checkpoints.
"""
import os
from os.path import join, dirname, exists
import gdown
import zipfile
from termcolor import colored

from cac.utils.file import repo_path

# download
force_download = False
print(colored("=> Downloading to assets/", 'yellow'))
url = 'https://drive.google.com/uc?id=1fkuQOEL3V7tDSMo0TvzKvWiMzuRjHgvd'
output = join(repo_path, 'assets/')
download_path = join(output, 'models-03-05-2021.zip')
if (not exists(download_path)) or force_download:
    download_path = gdown.download(url, output, quiet=False)
else:
    print(f"Zip file already exists at {download_path}")

# unzip
zip_file = download_path
unzip_dir = dirname(output)
print(colored(f"=> Unzipping {zip_file} to {unzip_dir}", 'yellow'))
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)
