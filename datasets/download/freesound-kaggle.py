"""
Script to download and setup FreeSound dataset

Kaggle link: https://www.kaggle.com/c/freesound-audio-tagging/data?select=train.csv)
Final dataset link: https://zenodo.org/record/2552860#.XFD05fwo-V4

Help:
>>> python freesound-kaggle.py
"""
from os import makedirs, symlink
from os.path import join, basename, exists, isdir
from glob import glob
from subprocess import call
from termcolor import colored
from tqdm import tqdm


DATA_DIR = '/data/freesound-kaggle/FSDKaggle2018'
makedirs(DATA_DIR, exist_ok=True)

URLs = [
	'https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1',
	'https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1',
	'https://zenodo.org/record/2552860/files/FSDKaggle2018.doc.zip?download=1',
	'https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip?download=1'
]

# download and unzip files at relevant location
for url in URLs:

	fname = basename(url.split('?download=1')[0])
	print(colored(f"=> Downloading {fname}", 'yellow'))

	zip_fpath = join(DATA_DIR, fname)
	unzip_folder = zip_fpath.split('.zip')[0]

	# download and unzip
	if not exists(zip_fpath):
		call(f"wget {url} -O {zip_fpath}", shell=True)

	if not isdir(unzip_folder):
		if 'train.zip' in fname:
			call(f"7z e {zip_fpath} -Y -o{unzip_folder}", shell=True)
		else:
			call(f"unzip {zip_fpath} -d {DATA_DIR}", shell=True)


# making symlinks in the raw/ folder
raw_folder = DATA_DIR.replace('FSDKaggle2018', 'raw')
makedirs(raw_folder, exist_ok=True)

audio_folder = join(raw_folder, 'audio')
makedirs(audio_folder, exist_ok=True)

# audio files
all_audio_files = glob(join(DATA_DIR, '*/*.wav'))
src_files = all_audio_files
dst_files = [join(audio_folder, basename(x)) for x in src_files]
for src, dst in tqdm(zip(src_files, dst_files), desc=colored('Creating symlinks', 'white')):
    if not exists(dst):
        symlink(src=src, dst=dst)

# other metadata
all_other_files = glob(join(DATA_DIR, '*.doc/*')) + glob(join(DATA_DIR, '*.meta/*'))
src_files = all_other_files
dst_files = [join(raw_folder, basename(x)) for x in src_files]
for src, dst in zip(src_files, dst_files):
    if not exists(dst):
        symlink(src=src, dst=dst)
