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
    
    median_days_cough = attributes[attributes['enroll_cough'] == 'Yes']['enroll_days_with_cough'].median()
    median_days_fever = attributes[attributes['enroll_fever'] == 'Yes']['enroll_days_with_fever'].median()
    median_days_shortness_of_breath = attributes[attributes['enroll_shortness_of_breath'] == 'Yes']['enroll_days_with_shortness_of_breath'].median()
    median_age = attributes.loc[attributes['enroll_patient_age'] < 100, 'enroll_patient_age'].median()

    attributes.loc[attributes['enroll_cough'] == 'No', 'enroll_days_with_cough'] = 0
    attributes.loc[attributes['enroll_fever'] == 'No', 'enroll_days_with_fever'] = 0
    attributes.loc[attributes['enroll_shortness_of_breath'] == 'No', 'enroll_days_with_shortness_of_breath'] = 0

    attributes['enroll_days_with_cough'].fillna(median_days_cough, inplace=True)
    attributes['enroll_days_with_fever'].fillna(median_days_fever, inplace=True)
    attributes['enroll_days_with_shortness_of_breath'].fillna(median_days_shortness_of_breath, inplace=True)

    attributes["enroll_patient_age"] = np.where(attributes["enroll_patient_age"] > 100, median_age, attributes['enroll_patient_age'])
    attributes["enroll_days_with_cough"] = np.where(attributes["enroll_days_with_cough"] > 100, median_days_cough, attributes['enroll_days_with_cough'])
    attributes["enroll_days_with_fever"] = np.where(attributes["enroll_days_with_fever"] > 100, median_days_fever, attributes['enroll_days_with_fever'])
    attributes["enroll_days_with_shortness_of_breath"] = np.where(attributes["enroll_days_with_shortness_of_breath"] > 100, median_days_shortness_of_breath, \
                                                                  attributes['enroll_days_with_shortness_of_breath'])

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
    


