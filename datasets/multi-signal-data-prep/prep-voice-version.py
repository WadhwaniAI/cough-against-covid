from cac.utils.io import read_yml, save_yml
import pandas as pd
import numpy as np
from os import makedirs
from os.path import join, dirname, basename, splitext
from librosa.core import get_duration
import argparse
from tqdm import tqdm

def get_user_from_path(path):
    filename = splitext(basename(path))[0]
    split_index = filename.find('_cough')
    user = filename[:split_index]
    user = '_'.join(user.split('_')[:-2])
    return user

def main(args):
    version = args.version
    data_root = '/data/wiai-facility'
    load_dir = join(data_root, 'processed/audio/')
    
    data_config = read_yml(join(f'/data/wiai-facility/processed/versions/{version}.yml'))
    attributes = pd.read_csv('/data/wiai-facility/processed/attributes.csv')
    
    keys = ['all', 'train', 'val', 'test']
    
    new_config = dict()
    for key in keys:
        print (f'Working on {key}')
        d = data_config[key]
        files = []
        labels = []
        starts = []
        ends = []
        users = []

        length = len(d['file'])
        for i in tqdm(range(length)):
            file = d['file'][i]
            label = d['label'][i]
            start = d['start'][i]
            end = d['end'][i]
            user = get_user_from_path(file)

            if user not in users:
                users.append(user)

                filename = splitext(basename(file))[0]
                split_index = filename.find('_cough')
                path = join(load_dir, ''.join([filename[:split_index], '_audio_1_to_10.wav']))
                end = get_duration(filename=path)

                files.append(path)
                labels.append(label)
                starts.append(start)
                ends.append(end)

        print (f'Number of files from {length} reduced to {len(ends)}')
        new_config[key] = {'end' : ends,
                           'file' : files,
                           'label' : labels,
                           'start' : starts}
        
    data_root = '/data/wiai-facility/processed/'
    save_version = f'{version}_voice'
    save_path = join(data_root, 'versions', f'{save_version}.yml')

    print (save_path)
    save_yml(new_config, save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input information')
    parser.add_argument('-v', '--version', default='default-clf', type=str,
                        help='Version File Name')
    args = parser.parse_args()
    main(args)
    