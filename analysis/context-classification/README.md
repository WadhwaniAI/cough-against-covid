## Instructions for Context Based Classification

To start with running context based models, we need to process the attributes file that is required to get the enrollment features like `enroll_patient_age`, `enroll_patient_temperature` etc. By processing, we mean label encoding the categorical features and normalizing the continuous ones. 

To directly do the above (processed attributes file from `/data/wiai-facilitiy/processed/attributes.csv` will be stored as `/data/wiai-facilitiy/processed/attributes_context_processed.csv`), 
1. Run the script `python process_attributes.py -a attributes` directly
2. Use the notebook `process_attributes.ipynb` to try out different stuff


