## Instructions for Context Based Classification

The data_prep section creates the necessary pickle files to run the context classification notebooks. There are two ways to get started here, 
### 1. Working on same splits
If you want to have the same splits as the ones that you created for the cough experiments, Run the `context-data-prep-script.py` script in the following manner (we want to create splits corresponding to `configs/default-clf.yml` using the attributes file `/data/wiai-facility/processed/attributes.csv`),
```bash
cd analysis/context-classification/data_prep
python context-data-prep-script.py -v default-clf -a attributes
```

### 2. Creating new splits
You want to create new splits from scratch and want to do this task independent of the cough experiments. Run the notebook `data_prep/create-new-splits.ipynb` to create the necessary pickle file.
