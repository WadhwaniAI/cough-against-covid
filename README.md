# Cough Against COVID-19
Code relase for the [Cough Against COVID-19](https://www.wadhwaniai.org/work/cough-against-covid/) Project by Wadhwani AI supported by the Gates' Foundation.

#### [Project page](https://www.wadhwaniai.org/work/cough-against-covid/) | [Code](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection) | [Paper (ArXiV)](https://arxiv.org/abs/2009.08790) | [Data](Coming-soon)

## Setup

We use docker to manage code dependencies. Please follow the steps [here](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection/setup) to set up all dependencies. This code works on both CPU-only machine/ GPU machine. However, it is recommended to use a GPU machine since CPU machine is very slow in runtime.

## Datasets

We use a combination of publicly-available datasets and our own collected datasets. Please follow the steps [here](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection/datasets) to download, process all datasets.

## Training

#### Training on existing datasets
To run training on datasets downloaded in previous step, please follow the steps [here](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection/training).

#### Training on your own datasets
In order to train on your own dataset(s), first, you need to set up the dataset following steps similar to those for existing dataset given [here](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection/datasets). This includes downloading/setting it in the right folder structure, processing and splitting (train-validation-test).
Next, you need to create a new `.yml` config file (like [this](https://github.com/WadhwaniAI/cough-against-covid/blob/pb/cough-detection/configs/experiments/covid-detection/v9_4_cough_adam_1e-4.yml)) and configure the dataset section:

```yml
dataset:
name: classification_dataset
config:
    - name: <name-of-your-dataset>
    version: <version-of-your-dataset>
```
You can also play around with various other hyperparameters in the config like optimizer, scheduler, batch sampler method, random crop duration, network architecture etc.

## Evaluation

#### Evaluating on existing datasets

#### Demo Google Collab notebook

