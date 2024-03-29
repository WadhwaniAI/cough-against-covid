## Evaluation

The following sections will have steps to evaluate a given model on a given dataset. As an example, we have used config `iclrw/cough/v9.7/adamW_1e-4.yml` but the steps hold for any other config. Please see details of configs used as part of our [ICLRW](../configs/experiments/iclrw/README.md) submission.

### Evaluating given checkpoints on existing datasets

Given the model checkpoint and corresponding config file, you can evaluate on a given dataset.
* Checkpoint: `assets/models/iclrw/cough/v9.7/adamW_1e-4/checkpoints/113_ckpt.pth.tar`
* Corresponding config: `configs/experiments/iclrw/cough/v9.7/adamW_1e-4.yml`
* Dataset: `wiai-facility | v9.7 | test`

1. Copy model checkpoint in appropriate output folder (run inside docker):
```bash
# copies from assets/models/ckpt_path/ to /output/experiments/ckpt_path/
ckpt_path=experiments/iclrw/cough/v9.7/adamW_1e-4/checkpoints/113_ckpt.pth.tar
python training/copy_model_ckpts.py -p $ckpt_path --dst_prefix experiments
```

2. Run forward pass and store metrics
```bash
cfg=iclrw/cough/v9.7/adamW_1e-4.yml
python evaluation/inference.py -v $cfg -e 113 -dn wiai-facility -dv v9.7 -m test --at softmax -t 0.1317
```
The results are published on the terminal with key metric being AUC-ROC. Here, explanation of args:

* `-v`: experiment version (config file)
* `-u`: corresponds to the user who trained the model,
        no need to pass this when you have config file in
        `configs/` folder.
* `-e`: epoch/checkpoint number of the trained model
* `-dn`: dataset name
* `-dv`: dataset version (name of `.yml` file stored)
* `-m`: mode, train/test/val
* `-at`: point of the outputs where aggregation is applied, e.g. after `softmax`
* `-t`: threshold at which the model is evaluated against at the given mode


#### Summary of ICLRW Experiments

| Model   | Dataset | Config file                                          | W&B link                                                                | Best val AUC/epoch/threshold | ILA threshold |
|---------|---------|------------------------------------------------------|-------------------------------------------------------------------------|------------------------------|---------------|
| Cough   | V9.4    | `experiments/iclrw/cough/v9.4/adamW_1e-4_cyclic.yml` | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/dl984dhd) | 0.6558/38/0.1565             | 0.2827        |
| Cough   | V9.7    | `experiments/iclrw/cough/v9.7/adamW_1e-4.yml`        | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/1ghxp8yb) | 0.6293/113/0.06858           | 0.1317        |
| Cough   | V9.8    | `experiments/iclrw/cough/v9.8/adamW_1e-4.yml`        | [Lnk](https://app.wandb.ai/wadhwani/cough-against-covid/runs/23e52em4)  | 0.789/47/0.1604              | 0.2170        |
| Context | V9.4    | `experiments/iclrw/context/v9.4/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/3hxu1xg7) | 0.6849/9/0.2339              |        0.2339 |
| Context | V9.7    | `experiments/iclrw/context/v9.7/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/3lhdao77) | 0.6054/31/0.2069             |        0.2069 |
| Context | V9.8    | `experiments/iclrw/context/v9.8/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/f6hlbm06) | 0.6484/44/0.2282             |        0.2282 |

> Note: W&B link may not work for you since it is within Wadhwani AI W&B account.


### Evaluating your trained models on existing datasets

Given you ran training with following config, you can run evaluation as follows:
* Corresponding config: `configs/experiments/iclrw/cough/v9.7/adamW_1e-4.yml`
* Dataset: `wiai-facility | v9.7 | test`

1. Run forward pass and store metrics
```bash
cfg=iclrw/cough/v9.7/adamW_1e-4.yml
python evaluation/inference.py -v $cfg -e 113 -dn wiai-facility -dv v9.7 -m test --at softmax
```

> **Note**: Here, you do not need to copy checkpoint since checkpoints are saved during training itself. Plus, you do not need to explicitly pass `-threshold` since it picks it up from validation set logs saved during training.

<!-- 
#### Evaluating a cough-based model checkpoint on a given dataset

**Task**: Evaluate model checkpoint `assets/models/covid-detection/v9_7_cough_adam_1e-4/checkpoints/192_ckpt.pth.tar` on dataset `wiai-facility`/version `v9.7`/ mode `test`. Note that the config corresponding to this checkpoint is `experiments/covid-detection/v9_7_cough_adam_1e-4.yml`.

**Steps**:
1. Copy model checkpoint in appropriate output folder (run inside docker):
```bash
# copies from assets/models/ckpt_path/ to /output/experiments/ckpt_path/
python training/copy_model_ckpts.py -p covid-detection/v9_7_cough_adam_1e-4/checkpoints/192_ckpt.pth.tar --dst_prefix experiments
```

2. Run forward pass and store metrics
```bash
cfg=experiments/covid-detection/v9_7_cough_adam_1e-4.yml
python evaluation/inference.py -v $cfg -e 192 -dn wiai-facility -dv v9.7 -m test --at softmax -t 0.3290
```
The results are published on the terminal with key metric being AUC-ROC. The results are at individual-level i.e. if a person has multiple audio files in the evaluaton dataset, predictions across those shall be aggregated by an aggregator function like `max`.

Here,
* `-v`: experiment version (config file)
* `-u`: corresponds to the user who trained the model,
        no need to pass this when you have config file in
        `configs/` folder.
* `-e`: epoch/checkpoint number of the trained model
* `-dn`: dataset name
* `-dv`: dataset version (name of `.yml` file stored)
* `-m`: mode, train/test/val
* `-at`: point of the outputs where aggregation is applied, e.g. after `softmax`
* `-t`: threshold at which the model is evaluated against at the given mode

### ICLR'21 Workshop Paper : Epoch and Checkpoint details
We have provided the [model](../configs/experiments/iclrw) checkpoints and threshold values for the ICLR'21 Workshop paper. To directly evaluate the models used in the paper without training, follow the steps mentioned above for any of the configs as shared at [link](../configs/experiments/iclrw).

<div align='center'>

|      | Cough Model<br>(Epoch / Threshold) | Context Model<br>(Epoch / Threshold) |
|------|:----------------------------------:|:------------------------------------:|
| v9.4 |             37 / 0.145             |              15 / 0.211              |
| v9.7 |             154 / 0.053            |              31 / 0.207              |
| v9.8 |             76 / 0.111             |              38 / 0.231              |

</div>

---

## Evaluating a trained model on a given dataset
**Task**: Evaluate a trained model with config file `experiments/covid-detection/v9_7_cough_adam_1e-4.yml` at epoch `192` on dataset `wiai-facility`/version `v9.7`/ mode `test`.

**Steps**:
1. Run forward pass and store metrics. Note that passing `-t` is not needed here since it will pick up the optimal threshold from validation set logs stored while training.
```bash
cfg=experiments/covid-detection/v9_7_cough_adam_1e-4.yml
python evaluation/inference.py -v $cfg -e 192 -dn wiai-facility -dv v9.7 -m test --at softmax
```
---

#### Evaluating a context-based model checkpoint on a given dataset

**Steps**:
1. Copy model checkpoint in appropriate output folder (run inside docker):
```bash
# copies from assets/models/ckpt_path/ to /output/experiments/ckpt_path/
python training/copy_model_ckpts.py -p iclrw/context/v9.7/context-neural/checkpoints/31_ckpt.pth.tar --dst_prefix experiments
```

2. Run forward pass and store metrics
```bash
cfg=experiments/iclrw/context/v9.7/context-neural.yml
python evaluation/inference.py -v $cfg -e 31 -dn wiai-facility -dv v9.7 -m test --at softmax -t 0.2069
```

#### Evaluating a context-based trained model on a given dataset

**Steps**:
1. Run forward pass and store metrics. Note that passing `-t` is not needed here since it will pick up the optimal threshold from validation set logs stored while training.
```bash
cfg=experiments/iclrw/context/v9.7/context-neural.yml
python evaluation/inference.py -v $cfg -e 31 -dn wiai-facility -dv v9.7 -m test --at softmax
``` -->

#### Evaluating an ensemble of cough-based and context-based model on a given dataset

1. Before running evaluation of ensemble of predictions, you need to run inference for the individual models. Follow aforementioned steps.

2. Create a meta config for ensembling models ([e.g.](../configs/experiments/iclrw/ensemble/cough_context_v9.7.yml)).
In this example, we are ensembling a cough-based model and context-based models with ensemling weights of 0.5 each.
```yaml
models:
  cough:
    version: experiments/iclrw/cough/v9.7/adamW_1e-4.yml
    epoch: 113
    user: null
    weight: 0.5
    agg_method: max

  context:
    version: experiments/iclrw/context/v9.7/context-neural.yml
    epoch: 31
    user: null
    weight: 0.5
    agg_method: max

data:
  mode: test
```

3. Run the ensembling to see result metrics
```bash
python evaluation/ensemble.py -c experiments/iclrw/ensemble/cough_context_v9.7.yml
```
