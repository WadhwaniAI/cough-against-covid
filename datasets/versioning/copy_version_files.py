"""
Copies dataset version files for all datasets from assets/data/ to common storage folder.
"""
from os.path import join, exists, isdir, basename
from glob import glob
from subprocess import call
from termcolor import colored

from cac.utils.file import repo_path

DATA_DIR = '/data'

assets_dir = join(repo_path, 'assets')
datasets = glob(join(assets_dir, 'data', '*'))


for dataset in datasets:
    print(colored(f"=> Copying version files for {basename(dataset)}", 'yellow'))

    src_dir = join(dataset, 'processed/versions')
    dst_dir = src_dir.replace(assets_dir, '')
    call(f'rsync -avzP {src_dir}/ {dst_dir}/',  shell=True)