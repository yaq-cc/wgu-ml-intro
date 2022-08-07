#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". You will need to use more features
features_list = [
    'poi',
    'salary',
    'bonus',
    'exercised_stock_options',
    'director_fees',
    'deferred_income',
    'from_messages',
    'restricted_stock',
    'from_poi_to_this_person',
    'long_term_incentive',
    'from_this_person_to_poi'
 ]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Feature Analysis
df = pd.DataFrame(data_dict).T
number_of_data_points = len(data_dict) # 146
count_of_nonpoi, count_of_poi = df.groupby('poi')['email_address'].count().tolist() # [128, 18]
number_of_features = len(df.columns) # 21 

dtypes = {
    'bonus': pd.Int32Dtype(),
    'deferral_payments': pd.Int32Dtype(),
    'deferred_income': pd.Int32Dtype(),
    'director_fees': pd.Int32Dtype(),
    'email_address': str,
    'exercised_stock_options': pd.Int32Dtype(),
    'expenses': pd.Int32Dtype(),
    'from_messages': pd.Int32Dtype(),
    'from_poi_to_this_person': pd.Int32Dtype(),
    'from_this_person_to_poi': pd.Int32Dtype(),
    'loan_advances': pd.Int32Dtype(),
    'long_term_incentive': pd.Int32Dtype(),
    'other': pd.Int32Dtype(),
    'poi': np.bool_,
    'restricted_stock': pd.Int32Dtype(),
    'restricted_stock_deferred': pd.Int32Dtype(),
    'salary': pd.Int32Dtype(),
    'shared_receipt_with_poi': pd.Int32Dtype(),
    'to_messages': pd.Int32Dtype(),
    'total_payments': pd.Int32Dtype(),
    'total_stock_value': pd.Int32Dtype()
}

### Convert string NaNs to np.nans
for name, features in data_dict.items():
    for feature in features:
        if features[feature] == 'NaN':
            features[feature] = np.nan

data_frame = pd.DataFrame(data_dict).T # Transposed
default_features = data_frame.columns.to_list() # Default Features
default_dtypes = [f.name for f in pd.DataFrame(data_dict).T.dtypes] #Default datatypes

### Imputed Features
data_frame = data_frame.astype(dtypes)
data_frame['null_counts_by_person'] = data_frame.isnull().sum(axis=1).sort_values(ascending=False)
data_frame_sparsity = (data_frame.count()/146).sort_values()

### Task 2: Remove outliers
drop = ('TOTAL', 'THE TRAVEL AGENCY IN THE PARK')
for name in drop:
    data_frame.drop(name, inplace=True)

df = data_frame[['poi', 'salary', 'bonus']].fillna(0)
# colors = {True: 'red', False: 'blue'}
# df.plot.scatter(x='salary', y='bonus', c=df['poi'].apply(lambda x: colors[x]), figsize=(12,8))

### Task 3: Create new feature(s)
data_frame['pct_email_from_poi'] = data_frame['from_poi_to_this_person'] / data_frame['from_messages']
data_frame['pct_email_to_poi'] = data_frame['from_this_person_to_poi'] / data_frame['to_messages']
### Store to my_dataset for easy export below.
my_dataset = data_frame.to_dict(orient="index")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)