'''
This file creates the version file necessary for running the multi-signal joint training. Basically repeats the voice version file thrice so that there is one to one correspondence
between the cough and voice version file. This is because we have 3 cough samples and 1 voice sample per patient. 
Sample Command
$ python prep-voice-multi-signal-version.py -v v9.8
Here, v9.8 is the cough version file corresponding to which we need to create the voice version file needed for multi-signal joint training.
'''
from cac.utils.io import read_yml, save_yml
import pandas as pd
import numpy as np
from os import makedirs
import os
import argparse
from os.path import join, dirname, basename, splitext
from tqdm import tqdm

def main(args):
    version = args.version
    keys = ['all', 'train', 'val', 'test']
    path = f'/data/wiai-facility/processed/versions/{version}_voice.yml'
    assert os.path.exists(path) == True, "Create Voice Version first using prep-voice-version.py"
         
    config = read_yml(path)
    
    new_config = dict()
    for key in tqdm(keys):
        df = pd.DataFrame(config[key])
        new_df = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)
        new_config[key] = {
            'end' : list(new_df['end'].values),
            'file' : list(new_df['file'].values),
            'label' : list(new_df['label'].values),
            'start' : list(new_df['start'].values)
        }
    data_root = '/data/wiai-facility/processed/'
    save_version = f'{version}_voice_repeated'
    save_path = join(data_root, 'versions', f'{save_version}.yml')
    print (f'path :{save_path}')

    save_yml(new_config, save_path)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input information')
    parser.add_argument('-v', '--version', default='default-clf', type=str,
                        help='Version File Name')
    args = parser.parse_args()
    main(args)

