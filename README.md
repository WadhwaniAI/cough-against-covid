# Cough Against COVID-19
Code relase for the [Cough Against COVID-19](https://www.wadhwaniai.org/work/cough-against-covid/) Project by the [Wadhwani Institute for Artificial Intelligence](https://www.wadhwaniai.org/) supported by the Gates' Foundation.

#### [Project page](https://www.wadhwaniai.org/work/cough-against-covid/) | [Code](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection) | [Paper (ArXiv)](https://arxiv.org/abs/2009.08790) | [Data](Coming-soon)
<!-- 
<object data="http://yoursite.com/the.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>
 -->
 ![Cough Against COVID-19](./assets/media/teaser-v6.png)


## Setup

We use docker to manage code dependencies. Please follow the steps [here](./setup) to set up all dependencies. This code works on both CPU-only machine/ GPU machine. However, it is recommended to use a GPU machine since CPU machine is very slow in runtime.

### Data version files
For the datasets used in this work, we create our own split files and those are released publicly. Please run the following (from inside docker container) to download them to `assets/data/` folder.

```bash
python setup/download_data_splits.py
```

### Pre-trained Models

Broadly, we release trained checkpoints for three kinds of models:

* Cough-based `ResNet-18` models for cough-detection
* Cough-based `ResNet-18` models for COVID-detection
* Context-based `TabNet` models for COVID-detection

Please run the following (from inside docker container) to download them to `assets/models/` folder.

## Datasets

We use a combination of publicly-available datasets and our own collected datasets. Please follow the steps [here](./datasets) to download, process all datasets.

## Demo notebooks

To try out our model(s) on sample data, please follow the instructions.

* **Cough-based model**: Follow the notebook [here](demo/covid_detection_from_cough.ipynb) to predict COVID from cough using a pretrained model released with the repository. If you want to try on your own cough samples, you can record and store them in `assets/data/` and run the notebook by changing appropriate paths.

* **Context-based model**: Follow the notebook [here](demo/covid_detection_from_contextual_data.ipynb) to predict COVID from contextual features like age, symptoms, travel history etc. If you want to try on your own contextual-features, you can modify the relevant cells and run the notebook.

## Training

#### Training on existing datasets
To run training on datasets downloaded in previous step, please follow the steps [here](https://github.com/WadhwaniAI/cough-against-covid/tree/pb/cough-detection/training).

#### Training any custom model on any given dataset

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

More coming soon!

## Evaluation

You can evaluate your own trained models or use released model checkpoints on a given dataset. Instructions for both of these are given [here](./evaluation/README.md).

#### Evaluating any custom model on any given dataset

Coming soon!

