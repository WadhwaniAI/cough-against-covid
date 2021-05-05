import pandas as pd
import os
from cac.utils.io import read_yml
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import argparse

def main(args):
    attributes = pd.read_csv(f'/data/wiai-facility/processed/{args.attribute}.csv')
    
    median_days = dict()
    symptoms = ['cough', 'fever', 'shortness_of_breath']
    for symptom in symptoms:
        median_days[symptom] = attributes[attributes[f'enroll_{symptom}'] == 'Yes'][f'enroll_days_with_{symptom}'].median()
        attributes.loc[attributes[f'enroll_{symptom}'] == 'No', f'enroll_days_with_{symptom}'] = 0
        attributes[f'enroll_days_with_{symptom}'].fillna( median_days[symptom], inplace=True)
        attributes[f'enroll_days_with_{symptom}'] = np.where(attributes[f'enroll_days_with_{symptom}'] > 100, median_days[symptom], \
                                                             attributes[f'enroll_days_with_{symptom}'])

    median_age = attributes.loc[attributes['enroll_patient_age'] < 100, 'enroll_patient_age'].median()
    attributes["enroll_patient_age"] = np.where(attributes["enroll_patient_age"] > 100, median_age, attributes['enroll_patient_age'])

    continuous_var = ['enroll_patient_age', 'enroll_patient_temperature', 'enroll_days_with_cough', 'enroll_days_with_shortness_of_breath', 'enroll_days_with_fever']
    categorical_var = ['enroll_travel_history', 'enroll_contact_with_confirmed_covid_case',
               'enroll_health_worker', 'enroll_fever', 'enroll_cough', 'enroll_shortness_of_breath']

    # Normalizing it
    for var in continuous_var:
        attributes[var] = (attributes[var] - attributes[var].mean()) / attributes[var].std()

    for var in categorical_var:    
        le = LabelEncoder()
        attributes[var] = le.fit_transform(attributes[var])
     
    file_name = 'attributes_context_processed'
    save_path = f'/data/wiai-facility/processed/{file_name}.csv'
    print (f'Saving process attributes file at : {save_path}')
    attributes.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input information')
    parser.add_argument('-a', '--attribute', default='attributes', type=str,
                        help='attribute file name')
    args = parser.parse_args()
    main(args)
    


