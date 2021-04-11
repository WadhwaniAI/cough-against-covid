### Training on existing datasets

> Note: This assumes you have setup dependencies as in `setup/README.md` and downloaded datasets as in `datasets/README.md`.

* Start docker container
```bash
bash create_container.sh -g <gpu-number> -n <container-name> -e <common-storage-folder> -u <your-user-folder> -p <port-number>

$ example: bash create_container.sh -g 0 -n sc-1 -e ~/cac/ -u piyush -p 8001
```

* Copy dataset version files from `assets/data/`
```bash
# run inside docker: copies from assets/data/* /data/*
$ python datasets/versioning/copy_version_files.py
```

* Copy cough-detection pre-trained model from `assets/models/`
```bash
# run inside docker: copies from assets/models/* /output/*
$ python training/copy_model_ckpts.py
```

* Run training, for example, with config file:
```bash
$ cfg=experiments/covid-detection/v9_4_cough_adam_1e-4.yml
$ python training/train.py -v $cfg --wandb_entity <your-W&B-account-name>
```
