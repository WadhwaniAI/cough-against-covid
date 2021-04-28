"""
Copies model checkpoints from assets/models/ to common storage folder.

NOTE: Run this script inside docker.
"""
import os
from os.path import join, exists, isdir, basename, dirname
import argparse
from glob import glob
from subprocess import call
from termcolor import colored

from cac.utils.file import repo_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copies model checkpoints")
    parser.add_argument('-a', '--copy_all', action='store_true',
                        help='copy all checkpoints from assets/models/')
    parser.add_argument('-p', '--ckpt_path', default=None, required=False,
                        help='path to ckpt to copy relative to assets/models/')
    args = parser.parse_args()

    assets_dir = join(repo_path, 'assets')
    models_dir = join(assets_dir, 'models')
    all_ckpts = glob(join(models_dir, '**/*.pth.tar'), recursive=True)


    if args.ckpt_path:
        src = join(assets_dir, 'models', args.ckpt_path)
        assert exists(src), f"Source ckpt file does not exist as {src}"
        dst = src.replace(models_dir, '/output/')
        print(colored(f"Copying from {src} to {dst}", 'yellow'))

        os.makedirs(dirname(dst), exist_ok=True)
        call(f'rsync -avzP {src} {dst}',  shell=True)

    else:
        if args.copy_all:
            for path in all_ckpts:
                src = path
                assert exists(src), f"Source ckpt file does not exist as {src}"
                dst = src.replace(models_dir, '/output')
                print(colored(f"Copying from {src} to {dst}", 'yellow'))

                os.makedirs(dirname(dst), exist_ok=True)
                call(f'rsync -avzP {src} {dst}',  shell=True)
