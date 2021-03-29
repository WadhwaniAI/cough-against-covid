"""
Script to download Wadhwani AI's Cough Against COVID dataset.
It is named as `wiai-facility/` because it is collected from
multiple testing facilities across India.
"""
import os
from os.path import join, exists, basename
from glob import glob
from subprocess import call
from termcolor import colored


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = '/data'
DATASET_DIR = create_folder(join(DATA_DIR, 'wiai-facility'))
# os.makedirs(DATASET_DIR, exist_ok=True)

# <------------- Code to download .zip files --------------->
# this shall be filled in when we have decided how to release
# the dataset. Please ask code owners to supply the dataset
# .zip file until it is offiicially released.
# command: rsync -avzP piyushb@192.168.100.70:/scratche/data/cac/data/wiai-facility/wiai-facility-23-11-2020.zip ./
# or  py7zr c -v 2000000k wiai-facility-23-11-2020.7z raw/
zip_fpath = join(DATASET_DIR, 'wiai-facility-23-11-2020.zip')
assert exists(zip_fpath), f"File {zip_fpath} not downloaded."
# <--------------------------------------------------------->

unzip_folder = join(DATASET_DIR, basename(zip_fpath).split('.zip')[0])
print(
    colored(f"Unzipping .zip files: {zip_fpath} -> {unzip_folder}", 'yellow')
)
call(f"unzip '{zip_fpath}' -d {unzip_folder}/", shell=True)

# link raw/ folder to raw files
raw_folder = create_folder(join(DATASET_DIR, 'raw'))
# os.makedirs(raw_folder, exist_ok=True)
print(
    colored("Linking raw/ -> raw files.", 'yellow')
)
audio_folder = create_folder(join(raw_folder, 'audio'))
src_audio_folders = glob(join(unzip_folder, 'raw/audio/patient_*'))
tgt_audio_folders = [x.replace(unzip_folder, DATASET_DIR) for x in src_audio_folders]

annot_folder = create_folder(join(raw_folder, 'annotations'))
src_annot_files = [join(unzip_folder, 'raw/annotations/CaC_label_sheet-final-nov23.csv')]
tgt_annot_files = [x.replace(unzip_folder, DATASET_DIR) for x in src_annot_files]
# os.makedirs(audio_folder, exist_ok=True)

