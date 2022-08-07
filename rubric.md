# Functionality
Code reflects the description in the answers to questions in the writeup. i.e. code performs the functions documented in the writeup and the writeup clearly specifies the final analysis strategy.

# Usability
poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.

# Data exploration
Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:

- total number of data points
- allocation across classes (POI/non-POI)
- number of features used
- are there features with many missing values? etc.

# Outlier Investigation (related lesson: "Outliers")

Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled.

# Pick an algorithm (related lessons: "Naive Bayes" through "Choose Your Own Algorithm")

At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.

# Discuss parameter tuning and its importance.

Response addresses what it means to perform parameter tuning and why it is important.

# Tune the algorithm (related lesson: "Validation")

	
At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

- GridSearchCV used for parameter tuning
- Several parameters tuned
- Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).

# Usage of Evaluation Metrics (related lesson: "Evaluation Metrics")

At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.

# Discuss validation and its importance.

Response addresses what validation is and why it is important.

# Validation Strategy (related lesson "Validation")

Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.

# Algorithm Performance

When tester.py is used to evaluate performance, precision and recall are both at least 0.3.

