"""
Script to download and properly store FluSense dataset (https://github.com/Forsad/FluSense-data)

Note: FluSense is nothing but a subset of the AudioSet dataset
released by Google with annotations by Al Hossain et al. The above
repository contains only annotations.

Example usage (from inside docker container):
>>> python flusense.py
"""

import argparse
import os
from os.path import join, exists, basename
from glob import glob
from tqdm import tqdm
from subprocess import call
from termcolor import colored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setting up FluSense dataset", allow_abbrev=False
    )
    parser.add_argument(
        "-p", "--dpath", default='/data/flusense/',
        help="Folder path where you want the dataset to be stored."
    )
    parser.add_argument(
        "-u", "--url", default='git@github.com:Forsad/FluSense-data.git',
        help="URL of the repository. Note: you will need to set SSH keys to use this OR pass HTML URL."
    )
    (args, unknown_args) = parser.parse_known_args()


    print(colored(f"1. CREATING DATASET FOLDER AT: {args.dpath}", 'yellow'))
    os.makedirs(args.dpath, exist_ok=True)

    print(colored("2. CLONING REPOSITORY ...", 'yellow'))
    repo_dir = join(args.dpath, 'FluSense-data')
    if not exists(repo_dir):
        call(f"git clone {args.url} {repo_dir}", shell=True)
    else:
        print(colored("-> REPOSITORY ALREADY EXISTS.", 'yellow'))

    # check if all zip files exists
    print(colored("3. UNZIPPING FOLDERS ...", 'yellow'))
    zip_fnames = [
        'FluSense audio-20210321T144002Z-002.zip',
        'FluSense audio-20210321T144002Z-004.zip',
        'FluSense audio-20210321T144002Z-001.zip',
        'FluSense audio-20210321T144002Z-003.zip'
    ]
    zip_files = [join(args.dpath, 'FluSense-data', x) for x in zip_fnames]
    unzipped_folder = join(args.dpath, "FluSense-data/FluSense audio/")
    wav_files = glob(join(unzipped_folder, '*.wav'))

    if len(wav_files) < 1171:
        for file in zip_files:
            assert exists(file), f"Zip file {file} missing. \
                Please see README to download AudioSet files from GDrive."
            unzip_folder = file.split('.zip')[0]
            call(f"unzip '{file}' -d {repo_dir}/", shell=True)

    print(colored("4. LINKING DATA TO RAW FOLDER ...", 'yellow'))

    # audio files
    os.makedirs(f"{args.dpath}/raw/audio/", exist_ok=True)
    src_files = glob(join(args.dpath, 'FluSense-data/FluSense audio/*.wav'))
    dst_files = [join(args.dpath, 'raw/audio', basename(x)) for x in src_files]
    for src, dst in tqdm(zip(src_files, dst_files), desc=colored("Creating symlinks for audios", "white")):
        if not exists(dst):
            os.symlink(src=src, dst=dst)

    # annotation files
    os.makedirs(f"{args.dpath}/raw/annotations/", exist_ok=True)
    src_files = glob(join(args.dpath, 'FluSense-data/flusense_data/*.TextGrid'))
    dst_files = [join(args.dpath, 'raw/annotations', basename(x)) for x in src_files]
    for src, dst in tqdm(zip(src_files, dst_files), desc=colored("Creating symlinks for annotations", "white")):
        if not exists(dst):
            os.symlink(src=src, dst=dst)

