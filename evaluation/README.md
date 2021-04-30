## Evaluation
### 1. Evaluating a model (not trained by the user) on a given checkpoint
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

#### ICLR'21 Workshop Paper : Epoch and Checkpoint details
We have provided the [model](../configs/experiments/iclrw) checkpoints and threshold values for the ICLR'21 Workshop paper. To directly evaluate the models used in the paper without training, follow the steps mentioned above for any of the configs as shared at [link](../configs/experiments/iclrw).

<div align='center'>

|      | Cough Model<br>(Epoch / Threshold) | Context Model<br>(Epoch / Threshold) |
|------|:----------------------------------:|:------------------------------------:|
| v9.4 |             37 / 0.145             |              15 / 0.211              |
| v9.7 |             154 / 0.053            |              31 / 0.207              |
| v9.8 |             76 / 0.111             |              38 / 0.231              |

</div>

---

### 2. Evaluating a trained model on a given dataset
**Task**: Evaluate a trained model with config file `experiments/covid-detection/v9_7_cough_adam_1e-4.yml` at epoch `192` on dataset `wiai-facility`/version `v9.7`/ mode `test`.

**Steps**:
1. Run forward pass and store metrics. Note that passing `-t` is not needed here since it will pick up the optimal threshold from validation set logs stored while training.
```bash
cfg=experiments/covid-detection/v9_7_cough_adam_1e-4.yml
python evaluation/inference.py -v $cfg -e 192 -dn wiai-facility -dv v9.7 -m test --at softmax
```

### 3. Evaluating an ensemble of cough-based and context-based model on a given dataset

1. Before running evaluation of ensemble of predictions, you need to run inference for the individual models. Follow aforementioned steps.

2. Create a meta config for ensembling models ([e.g.](../configs/experiments/ensemble/cough_context_v9.7.yml)).
In this example, we are ensembling a cough-based model and context-based models with ensemling weights of 0.5 each.
```yaml
models:
  cough:
    version: experiments/covid-detection/v9_7_cough_adam_1e-4.yml
    epoch: 192
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
python evaluation/ensemble.py -c experiments/ensemble/cough_context_v9.7.yml
```
