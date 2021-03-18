## Datasets: Downloading, cleaning and versioning

This section has code and instructions for downloading and setting up a dataset for usage (training, inference etc.). In this section, we describe the steps to download, clean and version a given dataset.

> Note: This assumes you have setup the dependencies by following the steps given in `setup/`. All the code you run in these steps should be run from inside a docker container.

<!-- 
For a given dataset, say our own dataset named `wiai-facility`, you need to follow the steps given below. As an example, we show instructions to download and setup the [CoSwara Dataset](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.
1. **Download**: You can follow the instructions in `download/wiai-facility/` to download and store the dataset in a folder structure as instructed.
2. **Cleaning**: For every dataset, we have a `raw/` folder that stores the audio files in the format of the original dataset. In order to standardise the storage, we keep a `processed/` folder where we store a flat-list of all audio files contained in `raw/`.
3. **Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. This is an important step and needs carefully attention. For example, if you want to split a dataset randomly into train, validation and test, you can use `XXX` notebook and save it as version `v1.0`. If you want to create a new (different) split of the *same* samples as in `v1.0`, then create `v1.1` and so on. If you decide to add more samples to the original dataset, then you should create `v2.0` and so on.

### Common storage folder

For storing datasets and model outputs for this project, please create a root folder on your machine (could be anywhere), for example,`/Users/piyushbagad/cac/`. Inside that, create the following structure: (you should replace `piyush` folder with a folder by your username - it could be anything).
```bash
cac/
    |--data/
    |--outputs/
    |____piyush/
```
-->

### Steps for dataset processing

For a given dataset, you need to follow the steps given below to be able to run training or inference on the dataset. As an example, we show instructions to download and setup the [CoSwara Dataset](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.

> Note: At this point, make sure you have fired up a docker container and you are running code inside it.

> TEMPORARY: Once you start the container, run in another terminal `git checkout pb/datasets` since dataset related code is in another branch.

* **Step 1: Download**: Run the following commands. Note that `/Users/piyushbagad/cac/data/` is being mounted to the container at `/data` and thus the dataset folder you pass can be `/data/coswara-15-03-21/`.
```bash
cd /workspace/cough-against-covid/datasets/download/
python coswara.py -p /data/coswara-15-03-21/
```
Check the data at `/data/coswara-15-03-21/`. The `raw/audio/` folder contains the audio files following the structure of the original dataset. In `raw/annotations/`, we store a CSV files containing the target labels and other metadata (like location, gender etc.).


* **Step 2: Cleaning**: In order to make is easy to work with multiple datasets, we standardise storage structure. We create a folder named `processed/` inside the dataset folder. First, we create a flat list (symlinks) of all audio files in the dataset in this folder. Second, we create two files `processed/attributes.csv` and `processed/annotations.csv` which contain the metadata attributes and target label(s) respectively. Thus, every dataset for this project expects this kind of a standard structure. For Coswara dataset, we use the following notebook to create achieve two things. (Note: See the `setup/README.md` file for instructions on how to fire up a Jupyter lab session from inside a docker container).
```bash
datasets/cleaning/coswara.ipynb
```

* **Step 3: Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. This is an important step and needs carefully attention. For example, if you want to split a dataset randomly into train, validation and test, you can use `XXX` notebook and save it as version `v1.0`. If you want to create a new (different) split of the *same* samples as in `v1.0`, then create `v1.1` and so on. If you decide to add more samples to the original dataset, then you should create `v2.0` and so on. If a given dataset has pre-specified splits, then we store them in `/data/<dataset-name>/processed/versions/default.yml`. For `coswara` dataset, refer to the following notebook to create a default version. You can use your own notebooks to create new versions.
```bash
datasets/versioning/coswara/default.yml
```

### Ready-to-use Datasets

WIP: This can contain a list of datasets with various statistics that we can release with our own dataset package - for these datasets, a user need not run these steps.