"""
Script to download and properly store Coswara dataset (https://github.com/iiscleap/Coswara-Data)

Example usage:
>>> python coswara.py -d /scratche/data/cac/data/coswara-15-03-21/
"""
import argparse
import os
from os.path import join, exists, basename
from glob import glob
from subprocess import call
from termcolor import colored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setting up Coswara dataset", allow_abbrev=False
    )
    parser.add_argument(
        "-p", "--dpath", required=True,
        help="Folder path where you want the dataset to be stored."
    )
    parser.add_argument(
        "-u", "--url", default='git@github.com:iiscleap/Coswara-Data.git',
        help="URL of the repository. Note: you will need to set SSH keys to use this OR pass HTML URL."
    )
    (args, unknown_args) = parser.parse_known_args()

    print(colored(f"1. CREATING DATASET FOLDER AT: {args.dpath}", 'yellow'))
    os.makedirs(args.dpath, exist_ok=True)

    print(colored("2. CLONING REPOSITORY ...", 'yellow'))
    repo_dir = join(args.dpath, 'Coswara-Data')
    if not exists(repo_dir):
        call(f"git clone {args.url} {repo_dir}", shell=True)
    else:
        print(colored("-> REPOSITORY ALREADY EXISTS.", 'yellow'))
    

    print(colored("3. EXTRACTING DATA ...", 'yellow'))
    call(f"cd {repo_dir}; python extract_data.py", shell=True)

    print(colored("4. LINKING DATA TO RAW FOLDER ...", 'yellow'))
    os.makedirs(f"{args.dpath}/raw/audio/", exist_ok=True)
    src_folders = glob(join(args.dpath, 'Coswara-Data/Extracted_data/', '202*'))
    dst_folders = [join(args.dpath, 'raw/audio', basename(x)) for x in src_folders]

    for src, dst in zip(src_folders, dst_folders):
        if not exists(dst):
            os.symlink(src=src, dst=dst)
    
    print(colored("5. MOVING DATASET SHEET ...", 'yellow'))
    files = ['combined_data.csv', 'csv_labels_legend.json']
    os.makedirs(join(args.dpath, 'raw', 'annotations'), exist_ok=True)
    for file in files:
        src = join(args.dpath, 'Coswara-Data', file)
        dst = join(args.dpath, 'raw', 'annotations', file)
        if not exists(dst):
            os.symlink(src, dst)
