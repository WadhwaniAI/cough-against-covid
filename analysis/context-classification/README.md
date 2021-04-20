## Instructions for Context Based Classification

There are two ways to get started here, 
### 1. Working on same splits
This step creates creates the pickle file necessary for running the classification notebooks. Run the `context-data-prep-script.py` script in the following manner (we want to create splits corresponding to `configs/default-clf.yml` using the attributes file `/data/wiai-facility/processed/attributes.csv`),
```bash
cd analysis/context-classification/data_prep
python context-data-prep-script.py -v default-clf -a attributes
```

### 2. Creating new splits
You want to create new splits from scratch and want to do this task independent of the cough experiments. Run the notebook `data_prep/create-new-splits.ipynb` to create the necessary pickle file.
