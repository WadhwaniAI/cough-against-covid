"""
Script to download ESC-50 dataset. (https://github.com/karolpiczak/ESC-50)
"""
import os
from os.path import join, exists, basename, dirname, isdir
from glob import glob
from tqdm import tqdm
from subprocess import call
from termcolor import colored


DATA_DIR = '/data/esc-50'
os.makedirs(DATA_DIR, exist_ok=True)


# download zipped file
print(colored("1. Downloding zip file ...", 'yellow'))
URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
zip_file = join(DATA_DIR, 'master.zip')
command = f"wget {URL} -O {zip_file}"
if not exists(zip_file):
	call(command, shell=True)

# unzip
print(colored("2. Unzipping ...", 'yellow'))
unzip_folder = join(DATA_DIR, 'ESC-50-master')
command = f"unzip {zip_file} -d {DATA_DIR}/"
if not isdir(unzip_folder):
	call(command, shell=True)

# create and fill raw/ folder
src_folder = join(DATA_DIR, 'ESC-50-master')

dst_folder = src_folder.replace('ESC-50-master', 'raw')
os.makedirs(dst_folder, exist_ok=True)

# link audios
print(colored("3. Symlinking  ESC-50-master/ with raw/ ...", 'yellow'))
src_audio_folder = join(src_folder, 'audio')
dst_audio_folder = join(dst_folder, 'audio')
os.makedirs(dst_audio_folder, exist_ok=True)
src_files = glob(join(src_audio_folder, '*.wav'))
dst_files = [join(dst_audio_folder, basename(x)) for x in src_files]
for src, dst in tqdm(zip(src_files, dst_files), desc=colored('Creating symlinks', 'white')):
    if not exists(dst):
        os.symlink(src=src, dst=dst)

# link metadata
src_meta_folder = join(src_folder, 'meta')
dst_meta_folder = join(dst_folder, 'meta')
os.makedirs(dst_meta_folder, exist_ok=True)
src_files = glob(join(src_meta_folder, '*'))
dst_files = [join(dst_meta_folder, basename(x)) for x in src_files]
for src, dst in tqdm(zip(src_files, dst_files), desc=colored('Creating symlinks', 'white')):
    if not exists(dst):
        os.symlink(src=src, dst=dst)
