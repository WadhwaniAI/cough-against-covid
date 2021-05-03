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
print(colored("=> Downloading to assets/models.zip", 'yellow'))
url = 'https://drive.google.com/uc?id=1Ev46K6NCzDWGdlHgJGYOBuC2ZQZI2RSy'
output = join(repo_path, 'assets/')
download_path = join(output, 'models.zip')
if not exists(download_path):
    gdown.download(url, output, quiet=False)
else:
    print("Zip file already exists at assets/models.zip")

# unzip
zip_file = join(output, 'models.zip')
unzip_dir = dirname(output)
print(colored(f"=> Unzipping {zip_file} to {unzip_dir}", 'yellow'))
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)
