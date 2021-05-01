### ICLR'21 Workshop Paper Configs

We have added the experiment configs that we used for the ICLR'21 [AI For Public Health](https://aiforpublichealth.github.io/) Workshop paper. We had three different splits on which we evaluated our models (`v9.4` : Time Based Split, `v9.7` : Site based split, `v9.8` : Random Split). The best configs for each of these splits across two input modalities (cough and context) are:

#### Context Model Configs
1. v9.4 : `experiments/iclrw/context/v9.4/context-neural.yml`
2. v9.7 : `experiments/iclrw/context/v9.7/context-neural.yml`
3. v9.8 : `experiments/iclrw/context/v9.8/context-neural.yml`

#### Cough Model Configs
1. v9.4 : `experiments/iclrw/cough/v9.4/adamW_1e-4_cyclic.yml`
2. v9.7 : `experiments/iclrw/cough/v9.7/adamW_1e-4.yml`
3. v9.8 : `experiments/iclrw/cough/v9.8/adamW_1e-4.yml`

#### Summary of Experiments

| Model   | Dataset | Config file                                          | W&B link                                                                | Best val AUC/epoch/threshold |
|---------|---------|------------------------------------------------------|-------------------------------------------------------------------------|------------------------------|
| Cough   | V9.4    | `experiments/iclrw/cough/v9.4/adamW_1e-4_cyclic.yml` | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/dl984dhd) | 0.6558/38/0.1565             |
| Cough   | V9.7    | `experiments/iclrw/cough/v9.7/adamW_1e-4.yml`        | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/1ghxp8yb) | 0.6293/113/0.06858           |
| Cough   | V9.8    | `experiments/iclrw/cough/v9.8/adamW_1e-4.yml`        | [Lnk](https://app.wandb.ai/wadhwani/cough-against-covid/runs/23e52em4)  | 0.789/47/0.1604              |
| Context | V9.4    | `experiments/iclrw/context/v9.4/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/3hxu1xg7) | 0.6849/9/0.2339              |
| Context | V9.7    | `experiments/iclrw/context/v9.7/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/3lhdao77) | 0.6054/31/0.2069             |
| Context | V9.8    | `experiments/iclrw/context/v9.8/context-neural.yml`  | [Link](https://app.wandb.ai/wadhwani/cough-against-covid/runs/f6hlbm06) | 0.6484/44/0.2282             |

> Note: W&B link may not work for you since it is within Wadhwani AI W&B account.
