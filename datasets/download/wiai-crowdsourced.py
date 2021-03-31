"""
Script to download Wadhwani AI's Cough Against COVID dataset (crowdsourced).
It is named as `wiai-crowdsourced/` because it is collected from
a global online crowdsourced campaign.
"""
import os
from os.path import join, exists, basename, isdir
from time import time
from glob import glob
from subprocess import call
from termcolor import colored
from tqdm import tqdm


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = '/data'
DATASET_DIR = create_folder(join(DATA_DIR, 'wiai-crowdsourced'))

# <------------- Download & unzip .zip files --------------->
"""
This shall be filled in when we have decided how to release
the dataset. Please ask code owners to supply the dataset
.zip file(s) until it is officially released.

(1) command: rsync -avzP piyushb@192.168.100.70:/scratche/data/cac/data/wiai-crowdsourced/wiai-crowdsourced-15-03-2021.zip ./

The zip file is of size around 235MBs. This takes about 2 mins of time.
"""
zip_fpath = join(DATASET_DIR, 'wiai-crowdsourced-15-03-2021.zip')
assert exists(zip_fpath), f'.zip file not downloaded at {zip_fpath}'

unzip_folder = zip_fpath.replace('wiai-crowdsourced-15-03-2021.zip', 'raw')
print(
    colored(f"=> Unzipping .zip files: {zip_fpath} -> {unzip_folder}", 'yellow')
)
if not isdir(unzip_folder):
    call(f'unzip {zip_fpath} -d {DATASET_DIR}', shell=True)
# <--------------------------------------------------------->

##### Note: the unzipped folder should already have the raw/ folder created.
##### So there is no need to create raw/ folder and link it to source folder.

raw_folder = join(DATASET_DIR, 'raw')
NUM_AUDIO_FOLDERS = 525
patient_folders = glob(join(raw_folder, 'audio/user_*'))
assert len(patient_folders) == NUM_AUDIO_FOLDERS


work_sheet = join(raw_folder, 'audio/CaC_work_sheet-mar15-anonymized.csv')
assert exists(work_sheet)
