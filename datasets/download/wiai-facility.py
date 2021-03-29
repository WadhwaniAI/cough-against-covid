"""
Script to download Wadhwani AI's Cough Against COVID dataset.
It is named as `wiai-facility/` because it is collected from
multiple testing facilities across India.
"""
import os
from os.path import join, exists, basename
from time import time
from glob import glob
from subprocess import call
from termcolor import colored
from tqdm import tqdm
import py7zr
import multivolumefile


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = '/data'
DATASET_DIR = create_folder(join(DATA_DIR, 'wiai-facility'))
# os.makedirs(DATASET_DIR, exist_ok=True)

# <------------- Code to download .zip files --------------->
"""
This shall be filled in when we have decided how to release
the dataset. Please ask code owners to supply the dataset
.zip file(s) until it is offiicially released.

(1) command: rsync -avzP piyushb@192.168.100.70:/scratche/data/cac/data/wiai-facility/wiai-facility-23-11-2020.zip ./
(2) rsync -avzP piyushb@192.168.100.70:/scratche/data/cac/data/wiai-facility/wiai-facility-23-11-2020.7z* ./

or  py7zr c -v 2000000k wiai-facility-23-11-2020.7z raw/ (takes about 40min per zip file.)
It takes about 40mins to unzip.
"""

use_7z = True
if use_7z:
    # extracting without creating intermediate large file
    zip_fpath = join(DATASET_DIR, 'wiai-facility-23-11-2020.7z')
    unzip_folder = join(DATASET_DIR, 'raw/')
    print(
        colored(f"=> Unzipping .zip files: {zip_fpath} -> {unzip_folder}", 'yellow')
    )
    if not exists(unzip_folder):
        start = time()
        with multivolumefile.open(zip_fpath, mode='rb') as target_archive:
            with py7zr.SevenZipFile(target_archive, 'r') as archive:
                archive.extractall(path=DATASET_DIR)
        end = time()
        print(colored(f"Took {end - start} seconds.", 'white'))

else:
    # usual zip file not supported
    raise NotImplementedError

# <--------------------------------------------------------->

##### Note: the unzipped folder should already have the raw/ folder created.
##### So there is no need to create raw/ folder and link it to source folder.

raw_folder = join(DATASET_DIR, 'raw')
NUM_PATIENTS = 7169
patient_folders = glob(join(raw_folder, 'audio/patient_*'))
assert len(patient_folders) == NUM_PATIENTS


label_sheet = join(raw_folder, 'annotations/CaC_label_sheet-final-nov23.csv')
assert exists(label_sheet)
