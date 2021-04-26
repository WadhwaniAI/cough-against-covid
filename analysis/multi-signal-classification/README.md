## Instructions for Multi-Signal Classification
##### NOTE: This section is specific to our dataset (wiai-facility)
In Multi-signal classification, we want to use different types of inputs (cough, speech and contextual data) and train them jointly on our covid classification task. To get started on running, we need to create the necessary attributes file (for contextual data) and version files (for speech data). The following sections will be needed to start running experiments.

### Attributes Preprocessing
##### Note : Ignore this step, if you created the `attributes_context_processed.csv` for the context classification models

To start with running multi-signal jointly trained models, we need to process the attributes file that is required to get the enrollment features like `enroll_patient_age`, `enroll_patient_temperature` etc. By processing, we mean label encoding the categorical features and normalizing the continuous ones. <br>

To directly do the above (processed attributes file from `/data/wiai-facilitiy/processed/attributes.csv` will be stored as `/data/wiai-facilitiy/processed/attributes_context_processed.csv`), 
- Run the script `python analysis/context-classification/process_attributes.py -a attributes` directly OR
- Use the notebook `analysis/context-classification/process_attributes.ipynb` to try out different stuff

### Data Preparation
We collect three cough samples and one voice (speech) sample from each patient. In order to jointly train them, we need to prepare version files so that they are paired accordingly. In short, we can just run, 
```bash
 python datasets/multi-signal-data-prep/prep-voice-version.py -v v9.8
 python datasets/multi-signal-data-prep/prep-voice-multi-signal-version.py -v v9.8
```
What the above two steps do for us,
1. Create the voice version file from the corresponding cough version file. This can be done using `python datasets/multi-signal-data-prep/prep-voice-version.py -v v9.8`. This will create the corresponding version voice file having the same splits as the cough version file. Since we have 1 voice sample for every 3 cough samples, the number of files in each split will go down by a factor of 3. This version file can also be used for voice based covid classification tasks.
2. Once the corresponding voice version file has been created (`v9.8_voice` from `v9.8`), we need to create a version where the cough and voice files are in sync, which means we need to repeat the voice files by a factor of 3 (As 3 cough files and 1 voice file for each patient). To do this, just run, `python datasets/multi-signal-data-prep/prep-voice-multi-signal-version.py -v v9.8` 

In order to explore more, there are alternative notebooks to see the underlying data preparation steps at [here](../datasets/multi-signal-data-prep)


### Training Multi-Signal Joint Trained Models
* Start docker container
```bash
bash create_container.sh -g <gpu-number> -n <container-name> -e <common-storage-folder> -u <your-user-folder> -p <port-number>

$ example: bash create_container.sh -g 0 -n sc-1 -e ~/cac/ -u piyush -p 8001
```

* Run training, for example, with config file:
```bash
$ cfg=experiments/multi-signal-training/naive/multi-signal-max.yml
$ python training/train.py -v $cfg --wandb_entity <your-W&B-account-name>
```

### Network Creation (Optional)

Networks are created in a different manner as opposed to single model classification configs. In order to provide more flexibility, you can create standard pytorch networks as created in this [link](../cac/models/multi_signal_models.py).


The configs under the [multi-signal-training](../configs/experiments/multi-signal-training) use the networks from [link](../cac/models/multi_signal_models.py). We have added a three basic network architures using different combinations of input(cough-context, cough-voice, cough-context-voice). All of them are naive non-learnt merging at the final layer before logit computation. 


##### New Network Architectures
In order to add new network architectures, a standard pytorch network needs to be added [here](../cac/models/multi_signal_models.py). Make sure the the inputs (`signals: List[torch.Tensor], context-signal: torch.Tensor`) to the forward function does not change. Once this has been created, the network name needs to passed in the config file under the `network` section, with the necessary init parameters under the `params` key. 
