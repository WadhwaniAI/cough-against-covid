"""
Script to upload dataset (`wiai-release`) from DGX to S3.

Note: this is purely for internal WIAI use only.

Usage:

1. Install basic dependencies in a virtualenv/conda environment
```bash
pip install tqdm
pip install awscli

# check version
aws --version
aws-cli/1.19.107 Python/3.7.3 Linux/4.15.0-124-generic botocore/1.20.107
```

2. Export your AWS credentials for the data-upload account
```bash
export AWS_ACCESS_KEY_ID=<access_key>
export AWS_SECRET_ACCESS_KEY=<secret_key>
```

3. Run the script to upload data
```bash
cd ./datasets/upload/
python wiai-release.py -d /scratche/data/cac/data/wiai-release-mini-spectrograms
```
"""

import sys
import os
from os.path import join, basename, dirname, isdir
import argparse
from glob import glob
from tqdm import tqdm


BUCKET_NAME = "wiai-cac-data-hosting"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Args for data upload script", allow_abbrev=False
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help="path to the dataset to be uploaded to S3.",
    )
    (args, unknown_args) = parser.parse_known_args()

    assert isdir(args.dataset_path), f"Dataset does not exist at {args.dataset_path}"

    local_folder = args.dataset_path
    remote_folder = basename(args.dataset_path)

    command = f'aws s3 sync {local_folder}/ s3://{BUCKET_NAME}/{remote_folder}/'

    os.system(command)


