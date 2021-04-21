## Model-based Analysis

This section has code (mostly notebooks) to analyze various aspects of a *trained* model. For example, visualizing learnt features at the penultimate layer etc.

### COVID Detection models

For each dataset, we have a set of notebooks each of which do a particular kind of analysis. For example, for `wiai-facility` dataset, we have the following two notebooks as examples:

* `analysis/covid-detection/wiai-facility/framework/attribute_based_analysis.ipynb`: This notebook is to check model performance across various attributes, like depending on whether a person has "cough" as symptom or not, how does the model performance vary?

* `analysis/covid-detection/wiai-facility/framework/instance_level_embeddings.ipynb`: In order to visualize tSNE embeddings of learnt features at the penultimate layer (or any other layer) conditioned on various attributes, check out this notebook.