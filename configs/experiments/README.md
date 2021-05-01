### Experiments
**1. [cough-detection](../configs/experiments/cough-detection)** : Cough detection configs that uses open source cough datasets using a ResNet18 backbones.

**2. [covid-detection](../configs/experiments/covid-detection)** : Configs in this folder perform the covid detection task 
- deep models on cough (`v9_4_cough_adam_1e-4.yml`) or voice (`v9.7_on voice_adam_5s.yml`) 
- classical models using contextual feature (`v9.7_symptoms_xgb.yml`), [AI4COVID](https://arxiv.org/abs/2004.01275v5) features (`v9.7_ai4covid_xgb.yml`) and [Cambridge](https://arxiv.org/pdf/2006.05919.pdf) features (`v9.7_cambridge_xgb.yml`) .

**3. [iclrw](../configs/experiments/iclrw)** : Cough based and context based Configs used in the ICLR'21 workshop paper. 

**4. [multi-signal-training](../configs/experiments/multi-signal-training)** : Some configs as a starting point to perform multi-signal classification on different kinds of multi-signal combinations (cough-context, cough-voice and cough-voice-context). For more details on Multi-Signal Model, refer to this [link](../../analysis/multi-signal-classification)

**5. [unsupervised](../configs/experiments/unsupervised)** : // Add Stuff // 
