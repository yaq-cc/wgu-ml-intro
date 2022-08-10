# %%
import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier

from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import matplotlib.pyplot as plt

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

# %%
with open('final_project_dataset.pkl', 'rb') as src:
    data_dict = pickle.load(src)

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

# Create new features
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

# Remove some meaningless entries
for name in ('TOTAL', 'THE TRAVEL AGENCY IN THE PARK'):
    data_dict.pop(name)

# Format data for ML activities
def format_and_split(data_dict, features_list):
    return targetFeatureSplit(
        featureFormat(
            data_dict,
            features_list
        )
    )
labels, features = format_and_split(data_dict, features_list)
labels, features = np.array(labels), np.array(features)



# %%
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


# %%
scaler = MinMaxScaler() # Changed from StandardScaler, MinMax 
# I dont think we can reuse this...
# selector = SelectKBest() # k, score_func

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

# {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

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

# %%
# # Python uses call by object reference.  
# # Estimators are objects; we can store them externally and 
# # reference them in the pipeline so that we can interact
# # directly with them outside of the pipeline.

# scaler = StandardScaler() # None
# selector = SelectKBest() # k, score_func
# # classifier = GaussianNB() # priors, var_smoothing
# classifier = RandomForestClassifier() # n_estimators, 

# pipeline = Pipeline([
#     ('scaler', scaler),
#     ('selector', selector),
#     ('classifier', classifier)
# ])

# param_grid = {
#     "selector__k": [2, 3, 4, 5, 6, 7, 8, 9],
#     "classifier__n_estimators": [105, 110, 115, 120, 125, 130, 135],
#     "classifier__criterion": ["gini", "entropy"],
#     "classifier__min_samples_split": [2, 3, 5]
# }

# scores = {
#     'precision': 0,
#     'accuracy': 0,
#     'recall': 0,
# }

# while scores['precision'] < 0.30 or scores['recall'] < 0.30:
#     clear_output(wait=True)
#     StratifiedShuffleSplit(n_splits=1, test_size=0.25).split
#     train_indices, test_indices = next(split(features, labels))
#     train_features, test_features = features[train_indices], features[test_indices]
#     train_labels, test_labels = labels[train_indices], labels[test_indices]

#     # Prepare training and testing data
#     param_search = GridSearchCV(pipeline, param_grid=param_grid)
#     best_params_ = param_search.fit(train_features, train_labels).best_params_ # {'selector__k': 5}
#     print(best_params_)
#     tuned_pipe = Pipeline([
#         ('scaler', scaler),
#         ('selector', selector.set_params(**param_extractor(best_params_, "selector"))),
#         ('classifier', classifier.set_params(**param_extractor(best_params_, "classifier")))
#     ])

#     tuned_pipe.fit(train_features, train_labels)
#     scores = scorer(test_labels, tuned_pipe.predict(test_features))
#     best_features = best_features_extractor(selector, features_list)
#     print("precision: ", scores['precision'], scores['precision'] < 0.30)
#     print("recall: ", scores['recall'],  scores['recall'] < 0.30)
#     print(scores['precision'] < 0.30 or scores['recall'] < 0.30)


