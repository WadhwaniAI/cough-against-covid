### Datasets: Downloading, cleaning and versioning

This section has code and instructions for downloading and setting up a dataset for usage (training, inference etc.). For a given dataset, say our own dataset named `wiai-facility`, you need to follow the steps given below. As an example, we show instructions to download and setup the [CoSwara Dataset](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.

1. **Download**: You can follow the instructions in `download/wiai-facility/` to download and store the dataset in a folder structure as instructed.
2. **Cleaning**: For every dataset, we have a `raw/` folder that stores the audio files in the format of the original dataset. In order to standardise the storage, we keep a `processed/` folder where we store a flat-list of all audio files contained in `raw/`.
3. **Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. This is an important step and needs carefully attention. For example, if you want to split a dataset randomly into train, validation and test, you can use `XXX` notebook and save it as version `v1.0`. If you want to create a new (different) split of the *same* samples as in `v1.0`, then create `v1.1` and so on. If you decide to add more samples to the original dataset, then you should create `v2.0` and so on.

### Common storage folder

For storing datasets and model outputs for this project, please create a root folder on your machine, for example,`/scratche/data/cac/`. Inside that, create the following structure: (you should replace `piyush` folder with a folder by your username - it could be anything).
```bash
cac/
    |--data/
    |--outputs/
    |____piyush/
```

### Applying the steps on sample dataset

For the purpose of testing, we use the [Coswara](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.

* **Step 1: Download**: Run the following commands. Note that depending on where you have created your root folder (e.g. `/scratche/data/cac/`), you should pass the folder for storing this dataset.
```bash
pip install termcolor
cd ~/projects/cac-test-release/datasets/download/
python coswara.py -p /scratche/data/cac/data/coswara-15-03-21/
```
Check the data at `/scratche/data/cac/data/coswara-15-03-21/`. The `raw/audio/` folder contains the audio files following the structure of the original dataset. In `raw/annotations/`, we store a CSV files containing the target labels and other metadata (like location, gender etc.).