## Model-based Analysis

This section has code (mostly notebooks) to analyze various aspects of a *trained* model. For example, visualizing learnt features at the penultimate layer etc.

### COVID Detection models

For each dataset, we have a set of notebooks each of which do a particular kind of analysis. For example, for `wiai-facility` dataset, we have the following two notebooks as examples:

* [Notebook](covid-detection/wiai-facility/framework/attribute_based_analysis.ipynb) to check model performance across various attributes, like depending on whether a person has "cough" as symptom or not, how does the model performance vary?

* [Notebook](covid-detection/wiai-facility/framework/instance_level_embeddings.ipynb) to visualize tSNE embeddings of learnt features at the penultimate layer (or any other layer) conditioned on various attributes etc.

---

## Model-free Analysis

This sections has code (mostly notebooks) to do unsupervised (or model-less) analysis of data, for example, considering classical features and visualizing them.

* [AI4COVID](unsupervised/wiai-facility/ai4covid.ipynb): This notebook has features proposed by [this](https://www.sciencedirect.com/science/article/pii/S2352914820303026) paper. You can visualize TSNE embeddings of features conditioned on various attributes.

* [Cambridge](unsupervised/wiai-facility/cambridge.ipynb): This notebook has features proposed by [this](https://arxiv.org/pdf/2006.05919.pdf) paper. You can visualize TSNE embeddings of features conditioned on various attributes.

---

## Data Analysis

To look at various distributions of data and summary statistics, refer to notebooks [here](data/wiai-facility/) for `wiai-facility` dataset. You may clone the notebooks and use them for other datasets as well by changing appropriate paths. 