## Datasets: Downloading, cleaning and versioning

This section has code and instructions for downloading and setting up a dataset for usage (training, inference etc.). In this section, we describe the steps to download, clean and version a given dataset.

> Note: This assumes you have setup the dependencies by following the steps given in `setup/`. All the code you run in these steps should be run from inside a docker container.

<!-- 
For a given dataset, say our own dataset named `wiai-facility`, you need to follow the steps given below. As an example, we show instructions to download and setup the [CoSwara Dataset](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.
1. **Download**: You can follow the instructions in `download/wiai-facility/` to download and store the dataset in a folder structure as instructed.
2. **Cleaning**: For every dataset, we have a `raw/` folder that stores the audio files in the format of the original dataset. In order to standardise the storage, we keep a `processed/` folder where we store a flat-list of all audio files contained in `raw/`.
3. **Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. This is an important step and needs carefully attention. For example, if you want to split a dataset randomly into train, validation and test, you can use `XXX` notebook and save it as version `v1.0`. If you want to create a new (different) split of the *same* samples as in `v1.0`, then create `v1.1` and so on. If you decide to add more samples to the original dataset, then you should create `v2.0` and so on.

### Common storage folder

For storing datasets and model outputs for this project, please create a root folder on your machine (could be anywhere), for example,`~/cac/`. Inside that, create the following structure: (you should replace `piyush` folder with a folder by your username - it could be anything).
```bash
cac/
    |--data/
    |--outputs/
    |____piyush/
```
-->

### Steps for dataset processing

In order to use a given dataset (training or inference), you need to follow the three steps as follows. 

* **Step 1 - Download**: For this step, we provide scripts to download and store a given dataset.

* **Step 2 - Cleaning**: In order to make is easy to work with multiple datasets, we standardise storage structure. For a given dataset, the structure looks as follows. On running the given scripts, dataset files will be stored in `DownloadedDatasetFiles`. The audio files shall be symlinked to `raw/audio/` and any annotations that come with the dataset shall be linked to `raw/annotations/`. In the cleaning step, we standardise s.t. `processed/audio/` will have a flat-list of audio files (note that `raw/audio/` may have any structure like `date/person-ID/*.wav`). `processed/attributes.csv` would contain the metadata associated with that dataset and `processed/annotations.csv` will contain the labels.

```bash
dataset/
├── DownloadedDatasetFiles
│   ├── XYZ
│   ├── ABC
│   └── :
├── processed
│   ├── audio/
│   ├── attributes.csv
│   ├── annotations.csv
│   └── versions/
└── raw
    ├── annotations/
    └── audio/
```

* **Step 3 - Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. For example, if you want to split a dataset randomly into train, validation and test, you can use our given notebooks and save it as version `v1.0` and so on. For sake of ease of use, we provide relevant data versions (splits) already in `assets/data/`  folder for various datasets. You can  copy these in the common storage folder as follows:
```bash
cd /workspace/cough-against-covid/datasets/versioning/
python copy_version_files.py
```
You can skip step `Versioning` from the processing for each dataset unless you wish to create a new split.

### Summary of Datasets

For this project, we use the following datasets. The instructions to follow steps 1, 2, 3 for each of them are given in the subsequent sections.

<div align="center">

| Dataset                                                                    | Cough | Non-cough | Total |
|----------------------------------------------------------------------------|-------|-----------|-------|
| [Coswara](https://github.com/iiscleap/Coswara-Data)                        | 3082  | 10786     | 13868 |
| [FluSense](https://github.com/Forsad/FluSense-data)                        | 2486  | 9201      | 11687 |
| [FreeSound](https://zenodo.org/record/2552860#.XFD05fwo-V4)                | 273   | 10800     | 11073 |
| [ESC-50](https://github.com/karolpiczak/ESC-50)                            | -     | 2000      | 2000  |
| [Ours (Wadhwani AI)](https://www.wadhwaniai.org/work/cough-against-covid/) | 11724 | 13429     | 25153 |

Number of COVID-19 positive/negative cough samples:

| Dataset                                                                    | Positive | Negative  | Unknown | Total |
|----------------------------------------------------------------------------|-------|------------|-------|-------|
| [Ours (Wadhwani AI)](https://www.wadhwaniai.org/work/cough-against-covid/) | 3867  | 7857       | - | 11724 |
| [Coswara](https://github.com/iiscleap/Coswara-Data)                        | 290   | 2426        | 366 | 3082 |


</div>

---
 
<!-- 
For a given dataset, you need to follow the steps given below to be able to run training or inference on the dataset. As an example, we show instructions to download and setup the [CoSwara Dataset](https://github.com/iiscleap/Coswara-Data) by IISc, Bangalore.

> Note: At this point, make sure you have fired up a docker container and you are running code inside it.

> TEMPORARY: All the code should be run on `pb/datasets` branch.

* **Step 1: Download**: Run the following commands. Note that `~/cac/data/` is being mounted to the container at `/data` and thus the dataset folder you pass can be `/data/coswara-15-03-21/`.
```bash
cd /workspace/cough-against-covid/datasets/download/
python coswara.py -p /data/coswara-15-03-21/
```
Check the data at `/data/coswara-15-03-21/`. The `raw/audio/` folder contains the audio files following the structure of the original dataset. In `raw/annotations/`, we store a CSV files containing the target labels and other metadata (like location, gender etc.).


* **Step 2: Cleaning**: In order to make is easy to work with multiple datasets, we standardise storage structure. We create a folder named `processed/` inside the dataset folder. First, we create a flat list (symlinks) of all audio files in the dataset in this folder. Second, we create two files `processed/attributes.csv` and `processed/annotations.csv` which contain the metadata attributes and target label(s) respectively. Thus, every dataset for this project expects this kind of a standard structure. For Coswara dataset, we use the following notebook to create achieve two things. (Note: See the `setup/README.md` file for instructions on how to fire up a Jupyter lab session from inside a docker container).
```bash
datasets/cleaning/coswara.ipynb
```

* **Step 3: Versioning**: Once steps 1 and 2 are done, you can use the code in `versioning/` to create dataset version files. This is an important step and needs carefully attention. For example, if you want to split a dataset randomly into train, validation and test, you can use `XXX` notebook and save it as version `v1.0`. If you want to create a new (different) split of the *same* samples as in `v1.0`, then create `v1.1` and so on. If you decide to add more samples to the original dataset, then you should create `v2.0` and so on. If a given dataset has pre-specified splits, then we store them in `/data/<dataset-name>/processed/versions/default.yml`. For `coswara` dataset, refer to the following notebook to create a default version for COVID-cough classification task. You can use your own notebooks to create new versions.
```bash
datasets/versioning/cough-classification/coswara/default.yml
```
 -->
## Setting up publicly available datasets

We use the following public datasets for pre-training our model before fine-tuning on the primary task(s). The pre-training task we consider is cough vs no-cough classification i.e. cough-detection.

### FreeSound Dataset (Kaggle)

* Download the dataset:
```bash
cd /workspace/cough-against-covid/datasets/download
python freesound-kaggle.py
```
* Clean (standardise) the dataset: Run the notebook - `cough-against-covid/datasets/cleaning/freesound-kaggle.ipynb`.
* Versioning (Optional): We use this dataset for cough-detection task only. Thus, we create version `v1.0` using notebook - `cough-against-covid/datasets/versioning/cough-detection/freesound-kaggle/v1.0.ipynb`. If you want to try a new split, you can create a new notebook.

The final dataset folder structure as a result of above steps is organized as follows:
```bash
freesound-kaggle/
├── FSDKaggle2018
│   ├── FSDKaggle2018.audio_test
│   ├── FSDKaggle2018.audio_train
│   ├── FSDKaggle2018.doc
│   └── FSDKaggle2018.meta
├── processed
│   ├── audio
│   └── versions
└── raw
    └── audio

11 directories
```

### Coswara Dataset

* Download: Run the following commands. Note that `~/cac/data/` is being mounted to the container at `/data` and thus the dataset folder you pass can be `/data/coswara-15-03-21/`.
```bash
cd /workspace/cough-against-covid/datasets/download/
python coswara.py -p /data/coswara-15-03-21/
```
Check the data at `/data/coswara-15-03-21/`. The `raw/audio/` folder contains the audio files following the structure of the original dataset. In `raw/annotations/`, we store a CSV files containing the target labels and other metadata (like location, gender etc.).

* Cleaning: Run the notebook - `cough-against-covid/datasets/cleaning/coswara.ipynb`.

* Versioning (Optional): Run the notebook -  `datasets/versioning/cough-detection/coswara/v1.0.ipynb`. Note that this version is for cough-detection. If you want to use Coswara for COVID classification, then you need to create a new version.

The final dataset folder structure as a result of above steps is organized as follows:
```bash
coswara-15-03-21/
├── Coswara-Data
│   └── Extracted_data
│       ├── 20200413
│       │   ├── 0Rlzhiz6bybk77wdLjxwy7yLDhg1
│       │   │   ├── breathing-deep.wav
│       │   │   ├── breathing-shallow.wav
│       │   │   ├── cough-heavy.wav
│       │   │   ├── cough-shallow.wav
│       │   │   ├── counting-fast.wav
│       │   │   ├── counting-normal.wav
│       │   │   ├── metadata.json
│       │   │   ├── vowel-a.wav
│       │   │   ├── vowel-e.wav
│       │   │   └── vowel-o.wav
│       │	└── :
│       └── :
├── processed
│   ├── audio
│   └── versions
└── raw
    ├── annotations
    └── audio

1570 directories
```

### FluSense Dataset

FluSense is nothing but a subset of the AudioSet dataset released by Google with annotations by [Al Hossain et al](https://github.com/Forsad/FluSense-data). This subset of AudioSet is stored as a google drive folder.

* Download: You need to download it from [here](https://drive.google.com/drive/folders/1c-qkb_ljD6xXqU4AGm4jEf8-lygRjLtS) and place it at this location: `/data/flusense/`. Unfortunately, this step needs to be done manually. After the `zip` files (4 in total) is downloaded from GDrive, run
```bash
mv /path/to/downloads/FluSense*.zip /path/to/common/storage/data/flusense/FluSense-data/
```
Once this is downloaded, run:
```bash
cd /workspace/cough-against-covid/datasets/download
python flusense.py
```

* Cleaning: Run the notebook - `cough-against-covid/datasets/cleaning/flusense.ipynb`.

* Versioning (Optional): We use this dataset for cough-detection task only. Thus, we create version `v1.0` using notebook - `cough-against-covid/datasets/versioning/cough-detection/flusense/segmented-v1.0.ipynb`. If you want to try a new split, you can create a new notebook.

The final dataset folder structure as a result of above steps is organized as follows:
```bash
flusense/
├── FluSense-data
│   ├── FluSense\ audio
│   └── flusense_data
├── processed
│   ├── audio
│   └── versions
└── raw
    ├── annotations
    └── audio

9 directories
```


Apart from the datasets for cough-detection, we also use ESC-50 dataset of environmental sounds for background noise addition (data augmentation) during training.

### ESC-50 Dataset

* Download: Run the following commands.
```bash
cd /workspace/cough-against-covid/datasets/download
python esc50.py
```

* Cleaning: Run the notebook - `cough-against-covid/datasets/cleaning/esc50.ipynb`.

* Versioning (Optional): We use this dataset for background noise addition. Thus, we create version `default` using notebook - `cough-against-covid/datasets/versioning/background/esc-50/default.ipynb`.


The final dataset folder structure as a result of above steps is organized as follows:
```bash
esc-50/
├── ESC-50-master
│   ├── audio
│   ├── meta
│   └── tests
├── processed
│   ├── audio
│   └── versions
└── raw
    ├── audio
    └── meta

10 directories
```

## Setting up COVID Datasets

### Cough Against COVID Dataset (Wadhwani AI)

Due to privacy constraints, we only provide spectrograms of all the audio that we have collected. To download/access the data, 
1. Sign the data sharing agreement [doc]()
2. Fill the form and attach the signed doc file 
3. Wait for approval, we generally respond in 2-3 days and longer if it's the weekend
4. Unzip the five zip files into a single folder (say zip_dir) that are provided on the request for data getting approved
5. Run the wiai-facility.py to unzip them into a single folder for further use

```bash
cd /workspace/cough-against-covid/datasets
python wiai-facility.py --zip_dir=/data/path_zip_file_dir/ --output_dir=/
 
Args:
    zip_dir: Directory where the zip files are present
    output_dir: Directory where the unzipped files would be present
```

This would unzip all the audio files at /storage/data/wiai-release-spectrograms/processed/spectrograms 

The final dataset folder structure as a result of above steps is organized as follows (only shows relevant files):
```bash
wiai-release-spectrograms/
    └── processed
        └── spectrograms
            ├── breathing_recording_7594558455454881112.npy
            ├── cough_sound_recording_1_8753244935701971117.npy
            ├── cough_sound_recording_2_82219589935506130.npy
            ├── cough_sound_recording_3_5125087475383634297.npy
            └── speech_recording_344728264604520646.npy

```
