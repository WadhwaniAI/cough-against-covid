### Evaluation

#### Evaluating a trained model on a given dataset

Suppose you trained a model with config file `experiments/covid-detection/v9_7_cough_adam_1e-4.yml`. To evaluate this model on dataset `wiai-facility`/version `v9.7`/ mode `test`, run the following:
```bash
cfg=experiments/covid-detection/v9_7_cough_adam_1e-4.yml
python inference.py -v $cfg -u piyush -e 1 -dn wiai-facility -dv v9.7 -m val --at softmax
```
The results are published on the terminal with key metric being AUC-ROC. The results are at individual-level i.e. if a person has multiple audio files in the evaluaton dataset, predictions across those shall be aggregated by an aggregator function like `max`.

Here,

* `-v`: experiment version (config file)
* `-u`: corresponds to the user who trained the model, 
        in my case, model checkpoints would be saved at
        `~/cac/outputs/piyush/` and thus I'd pass `piyush`.
* `-e`: epoch/checkpoint number of the trained model
* `-dn`: dataset name
* `-dv`: dataset version (name of `.yml` file stored)
* `-m`: mode, train/test/val
* `-at`: point of the outputs where aggregation is applied, e.g. after `softmax`


#### Evaluating any custom model on a given dataset

Coming soon!