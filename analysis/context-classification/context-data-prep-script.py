import pandas as pd
import os
from cac.utils.io import read_yml
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import pickle
import argparse

def get_patient_id(file):
    s = file.split('/')[-1].split('_')
    patient_id = '_'.join([s[0], s[1]])
    return patient_id

def main(args):
    version = args.version
    attributes = pd.read_csv(f'/data/wiai-facility/processed/{args.attribute}.csv')

    mean_days_cough = np.mean(attributes[attributes['enroll_cough'] == 'Yes']['enroll_days_with_cough'])
    mean_days_fever = np.mean(attributes[attributes['enroll_fever'] == 'Yes']['enroll_days_with_fever'])
    mean_days_shortness_of_breath = np.mean(attributes[attributes['enroll_shortness_of_breath'] == 'Yes']['enroll_days_with_shortness_of_breath'])
    median_age = attributes.loc[attributes['enroll_patient_age'] < 100, 'enroll_patient_age'].median()
    
    # variables to be considered
    cols = ['enroll_patient_age', 'enroll_days_with_fever', 'enroll_days_with_cough', 'enroll_days_with_shortness_of_breath',
               'enroll_patient_temperature', 'enroll_travel_history', 'enroll_contact_with_confirmed_covid_case',
            'enroll_health_worker', 'enroll_fever', 'enroll_cough',  'enroll_shortness_of_breath']

    config = read_yml(f'/data/wiai-facility/processed/versions/{version}.yml')
    sets = ['train', 'val', 'test']

    train_df = pd.DataFrame(config['train'])
    val_df = pd.DataFrame(config['val'])
    test_df = pd.DataFrame(config['test'])
    
    train_df['patient_id'] = train_df.file.apply(lambda x : get_patient_id(x))
    val_df['patient_id'] = val_df.file.apply(lambda x : get_patient_id(x))
    test_df['patient_id'] = test_df.file.apply(lambda x : get_patient_id(x))

    # sanity check
    print (f'------------------- Performing Sanity Check -------------------')
    patient_ids = train_df.patient_id.values
    unique = np.unique(patient_ids)
    assert 3 * len(unique) == len(train_df)

    print (f'------------------- Merging -------------------')
    attributes = attributes[['patient_id'] + cols]
    merged_train_df = train_df.merge(right=attributes, how='inner', on='patient_id')
    merged_val_df = val_df.merge(right=attributes, how='inner', on='patient_id')
    merged_test_df = test_df.merge(right=attributes, how='inner', on='patient_id')

    merged_train_df = merged_train_df[['patient_id', 'label'] + cols]
    merged_val_df = merged_val_df[['patient_id', 'label'] + cols]
    merged_test_df = merged_test_df[['patient_id', 'label'] + cols]

    mean_days = {'enroll_days_with_cough' : mean_days_cough, 'enroll_days_with_fever': mean_days_fever,
                 'enroll_days_with_shortness_of_breath' : mean_days_shortness_of_breath}

    merged_train_df.loc[merged_train_df['enroll_cough'] == 'No', 'enroll_days_with_cough'] = 0
    merged_val_df.loc[merged_val_df['enroll_cough'] == 'No', 'enroll_days_with_cough'] = 0
    merged_test_df.loc[merged_test_df['enroll_cough'] == 'No', 'enroll_days_with_cough'] = 0
    merged_train_df['enroll_days_with_cough'].fillna(mean_days['enroll_days_with_cough'], inplace=True)
    merged_val_df['enroll_days_with_cough'].fillna(mean_days['enroll_days_with_cough'], inplace=True)
    merged_test_df['enroll_days_with_cough'].fillna(mean_days['enroll_days_with_cough'], inplace=True)

    merged_train_df.loc[merged_train_df['enroll_fever'] == 'No', 'enroll_days_with_fever'] = 0
    merged_val_df.loc[merged_val_df['enroll_fever'] == 'No', 'enroll_days_with_fever'] = 0
    merged_test_df.loc[merged_test_df['enroll_fever'] == 'No', 'enroll_days_with_fever'] = 0
    merged_train_df['enroll_days_with_fever'].fillna(mean_days['enroll_days_with_fever'], inplace=True)
    merged_val_df['enroll_days_with_fever'].fillna(mean_days['enroll_days_with_fever'], inplace=True)
    merged_test_df['enroll_days_with_fever'].fillna(mean_days['enroll_days_with_fever'], inplace=True)

    merged_train_df.loc[merged_train_df['enroll_shortness_of_breath'] == 'No', 'enroll_days_with_shortness_of_breath'] = 0
    merged_val_df.loc[merged_val_df['enroll_shortness_of_breath'] == 'No', 'enroll_days_with_shortness_of_breath'] = 0
    merged_test_df.loc[merged_test_df['enroll_shortness_of_breath'] == 'No', 'enroll_days_with_shortness_of_breath'] = 0
    merged_train_df['enroll_days_with_shortness_of_breath'].fillna(mean_days['enroll_days_with_shortness_of_breath'], inplace=True)
    merged_val_df['enroll_days_with_shortness_of_breath'].fillna(mean_days['enroll_days_with_shortness_of_breath'], inplace=True)
    merged_test_df['enroll_days_with_shortness_of_breath'].fillna(mean_days['enroll_days_with_shortness_of_breath'], inplace=True)

    merged_train_df["enroll_patient_age"] = np.where(merged_train_df["enroll_patient_age"] > 100, median_age, merged_train_df['enroll_patient_age'])
    merged_val_df["enroll_patient_age"] = np.where(merged_val_df["enroll_patient_age"] > 100, median_age, merged_val_df['enroll_patient_age'])
    merged_test_df["enroll_patient_age"] = np.where(merged_test_df["enroll_patient_age"] > 100, median_age, merged_test_df['enroll_patient_age'])

    # sanity check 
    print (f'sanity check to make sure all values are present')
    print (f'train : {merged_train_df.isna().sum()}')
    print (f'val : {merged_val_df.isna().sum()}')
    print (f'test : {merged_test_df.isna().sum()}')

    dest_path = '/data/wiai-facility/processed/'
    file_name = f'df_{version}.pickle'
    
    df_dict = {'train' : merged_train_df, 'val' : merged_val_df, 'test': merged_test_df}
    final_path = os.path.join(dest_path, file_name)
    
    print (f'Saving a dictionary with both train, val and test dataframes at {final_path}')
    with open(final_path, 'wb') as handle:
        pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take input information')
    parser.add_argument('-v', '--version', required=True, type=str,
                        help='version of the experiment config file')
    parser.add_argument('-a', '--attribute', default='attributes_nov23', type=str,
                        help='attribute file name')
    args = parser.parse_args()
    main(args)
    


