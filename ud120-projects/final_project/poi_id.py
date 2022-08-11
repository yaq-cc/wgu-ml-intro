#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))
import warnings
warnings.filterwarnings("ignore")

from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". You will need to use more features

features_list = [
    'poi',
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'from_messages',
    'from_poi_to_this_person',
    'from_ratio',
    'from_this_person_to_poi',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi',
    'to_messages',
    'to_ratio',
    'total_payments',
    'total_stock_value'
]


### Outlier Analysis

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


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

### Outlier pruning - visualization
# df = data_frame[['poi', 'salary', 'bonus']].fillna(0)
# colors = {True: 'red', False: 'blue'}
# df.plot.scatter(x='salary', y='bonus', c=df['poi'].apply(lambda x: colors[x]), figsize=(12,8))

### Task 2: Remove outliers
drop = ('TOTAL', 'THE TRAVEL AGENCY IN THE PARK')
for name in drop:
    data_frame.drop(name, inplace=True)

### Outlier pruning - visualization (v2)
# df = data_frame[['poi', 'salary', 'bonus']].fillna(0)
# colors = {True: 'red', False: 'blue'}
# df.plot.scatter(x='salary', y='bonus', c=df['poi'].apply(lambda x: colors[x]), figsize=(12,8))


### Restart data from source to adhere to project requirements.

with open('final_project_dataset.pkl', 'rb') as src:
    data_dict = pickle.load(src)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### This list of features is the initial set of features representing
### all the features in the data set - later, we'll use SelectKBest 
### to optimize the feature set.

feature_set = set()

# find all unique feature
data_items = data_dict.items()
for employee, features in data_items:
    for feature in features:
        feature_set.add(feature)

# validate that all records have those features.
for employee, features in data_items:
    contains_all_features = all([f in feature_set for f in features.keys()])
    if not contains_all_features:
        raise KeyError("features missing...")

feature_set.remove('poi')
feature_set.remove('email_address') # feature will not work in featureFormat

### Task 2: Remove outliers
for name in ('TOTAL', 'THE TRAVEL AGENCY IN THE PARK'):
    data_dict.pop(name)

### Task 3: Create new feature(s)
for emp in data_dict:
    if not data_dict[emp]['from_poi_to_this_person'] == 'NaN' \
        and data_dict[emp]['from_this_person_to_poi'] == 'NaN':
        data_dict[emp]['from_ratio'] = data_dict[emp]['from_poi_to_this_person'] / data_dict[emp]['to_messages']
        data_dict[emp]['to_ratio'] = data_dict[emp]['from_this_person_to_poi'] / data_dict[emp]['from_messages']
    else:
        data_dict[emp]['from_ratio'] = 'NaN'
        data_dict[emp]['to_ratio'] = 'NaN'

feature_set.add('from_ratio')
feature_set.add('to_ratio')
set_list = list(feature_set)
set_list.sort()
features_list = ['poi'] + set_list


### Extract features and labels from dataset for local testing
# Format data for ML activities
def format_and_split(data_dict, features_list):
    dataset = targetFeatureSplit(
        featureFormat(
            data_dict,
            features_list
        )
    )
    return [
        np.array(data) 
        for data in dataset
    ]

labels, features = format_and_split(data_dict, features_list)

### Helper functions 
def param_extractor(best_params_, estimator_name):
    """ 
    param_extractor extracts and formats the best hyper-parameters
    from the best_params_ attribute of GridSearchCV
    """

    if not estimator_name.endswith("__"):
        estimator_name = estimator_name + "__"
    
    return {
    k.replace(estimator_name, ""): v
    for k, v in best_params_.items()
    if k.startswith(estimator_name)
    }

def scorer(test_labels, test_predictions):
    """
    scorer generates accuracy, precision, and recall
    metrics for a given model
    """
    return dict(
        accuracy=accuracy_score(test_labels, test_predictions),
        precision=precision_score(test_labels, test_predictions),
        recall=recall_score(test_labels, test_predictions),
    )

def best_features_extractor(selector, features_list):
    best_features = []
    for kbest in selector.get_support(indices=True):
        best_features.append(features_list[1:][kbest])
    return best_features

def set_estimator_params(pipeline, best_params_):
    # Update selector params
    pipeline.steps[1][1].set_params(**param_extractor(best_params_, 'selector'))
    pipeline.steps[2][1].set_params(**param_extractor(best_params_, 'classifier'))



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, accuracy_score

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


scaler = MinMaxScaler() # Changed from StandardScaler, MinMax 
gaussian = GaussianNB() # priors, var_smoothing
random_forest = RandomForestClassifier() # n_estimators, 
neighbors = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()
ada_boost = AdaBoostClassifier()

estimators = [{
	'name': "GaussianNB",
	'classifier': Pipeline([
		('scaler', scaler),
		('selector', SelectKBest()),
		('classifier', gaussian)
	]),
	'param_grid': {
		"selector__k": [5, 7, 9],
		"classifier__var_smoothing": np.logspace(0,-9, num=20),
	}
}, {
	'name': 'RandomForestClassifier',
	'classifier': Pipeline([
		('scaler', scaler),
		('selector', SelectKBest()),
		('classifier', random_forest)
	]),
	'param_grid': {
		"selector__k": [5, 7, 9],
		"selector__score_func": [f_classif, mutual_info_classif, chi2],
		"classifier__n_estimators": [110, 115, 120, 125],
		"classifier__criterion": ["gini", "entropy"],
		"classifier__min_samples_split": [2, 3, 5],
	}
}, {
	'name': 'KNeighborsClassifier',
	'classifier': Pipeline([
		('scaler', scaler),
		('selector', SelectKBest()),
		('classifier', neighbors)
	]),
	'param_grid': {
		"selector__k": [5, 7, 9],
		"selector__score_func": [f_classif, mutual_info_classif, chi2],
		"classifier__n_neighbors": [3, 5, 7, 9],
		"classifier__weights": ['uniform', 'distance'],
		"classifier__algorithm": ['ball_tree', 'kd_tree', 'brute'],
		"classifier__metric": ['minkowski', 'cityblock', 'euclidean'],
	}
}, {
	'name': 'DecisionTreeClassifier',
	'classifier': Pipeline([
		('scaler', scaler),
		('selector', SelectKBest()),
		('classifier', decision_tree)
	]),
	'param_grid': {
		"selector__k": [3, 5, 7, 9],
		"selector__score_func": [f_classif, mutual_info_classif, chi2],
		"classifier__max_leaf_nodes": [2, 3, 5, 8, 13],
		"classifier__min_samples_split": [2, 3, 4, 5, 6, 7],
	}
}, {
	'name': 'AdaBoostClassifier',
	'classifier': Pipeline([
		('scaler', scaler),
		('selector', SelectKBest()),
		('classifier', ada_boost)
	]),
	'param_grid': {
		"selector__k": [3, 5, 7, 9],
		"selector__score_func": [f_classif, mutual_info_classif, chi2],
		"classifier__n_estimators": [50, 63, 75, 86, 100],
		"classifier__learning_rate": [0.50, 0.75, 1, 1.25, 1.5],
	}
}]

# Evaluate various classifiers
print("Evaluating various classifiers.")
for index, estimator in enumerate(estimators):
	classifier = estimator['classifier']
	grid = estimator['param_grid']
	n_splits = 10
	splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25).split
	scores = {
		'accuracy': [],
		'precision': [],
		'recall': [],
	}
	for train_indices, test_indices in splitter(features, labels):
		train_features, test_features = features[train_indices], features[test_indices]
		train_labels, test_labels = labels[train_indices], labels[test_indices]
		pipe.fit(train_features, train_labels)
		score = scorer(test_labels, pipe.predict(test_features))
		scores['accuracy'].append(score['accuracy'])
		scores['precision'].append(score['precision'])
		scores['recall'].append(score['recall'])
	mean_scores = {
		'accuracy': sum(scores['accuracy']) / n_splits,
		'precision': sum(scores['precision']) / n_splits,
		'recall': sum(scores['recall']) / n_splits,
	}
	estimators[index].update({
		'mean_scores': mean_scores
	})
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    ### Tuning has been integrated into my pipeline :-)

	param_search = GridSearchCV(classifier, param_grid=grid)
	best_params_ = param_search.fit(train_features, train_labels).best_params_
	estimators[index].update({
		"best_params_": best_params_
	})
	estimators[index].update({
		'best_estimator_': param_search.best_estimator_
	})
	estimators[index].update({
		'best_features': best_features_extractor(param_search.best_estimator_.steps[1][1], features_list),
		'best_features_mask': param_search.best_estimator_.steps[1][1].get_support() 
	})
	print("Estimator: ", estimator['name'], "scores: ", mean_scores)

print("Performing synthetic tests with stratified sampling.")
for index, estimator in enumerate(estimators):
    pipeline = estimator['best_estimator_']
    scores = {
      'accuracy': [],
      'precision': [],
      'recall': [],
    }
    n_splits = 100
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25).split
    for train_indices, test_indices in splitter(features, labels):
        train_features, test_features = features[train_indices], features[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]
        pipeline.fit(train_features, train_labels)
        score = scorer(test_labels, pipeline.predict(test_features))
        scores['accuracy'].append(score['accuracy'])
        scores['precision'].append(score['precision'])
        scores['recall'].append(score['recall'])
    mean_scores = {
      'accuracy': sum(scores['accuracy']) / n_splits,
      'precision': sum(scores['precision']) / n_splits,
      'recall': sum(scores['recall']) / n_splits,
	  }
    print(estimator['name'], mean_scores, estimator['best_features'])

# Exclusively for analysis and validation purposes.
for estimator in estimators:
    classifier = estimator['best_estimator_']
    best_features = estimator['best_features']
    
    test_classifier(
        classifier, 
        data_dict, 
        feature_list=['poi'] + best_features
    )


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf, my_dataset, features_list = load_classifier_and_data()

# dump_classifier_and_data(clf, my_dataset, features_list)
# test_classifier performance:
# Pipeline(memory=None,
#      steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('selector', SelectKBest(k=5, score_func=<function f_classif at 0x7feb5dd634d0>)), ('classifier', GaussianNB(priors=None, var_smoothing=1.0))])
# 	Accuracy: 0.87571	Precision: 0.62264	Recall: 0.33000	F1: 0.43137	F2: 0.36424
# 	Total predictions: 14000	True positives:  660	False positives:  400	False negatives: 1340	True negatives: 11600