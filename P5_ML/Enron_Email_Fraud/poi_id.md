
# Machine learning to analyze Enron Dataset
**Udacity Nanodegree P5 project**

*Author: Jie Hu,  jie.hu.ds@gmail.com*

------------

## 0. Abstract

This is a project in which I use skills learned from Udacity course, including data wrangling, exploratory data analysis and machine learning, to do research on [Enron Fraud Email dataset](https://www.cs.cmu.edu/~./enron/).

The goal of this research is to find out most significant features to predict whether a person in the dataset is committed to fraud. The structure of this article is:
- Data Wrangling, in which I modify NA values and remove outliers
- Feature Selecting, in which I create some features I think important to predict fraud
- Training and tuning machine learning, in which I use sklearn to train 4 different models and compare their performance matrices, including precision, recall and f1 score
- Final part, in which I select Naive Bayes as my best model
- Conclusion

## 1. Data Wrangling

Firstly, load the dataset:


```python
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

```


```python
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
```

The dataset has 146 observations (people) including 18 person of interest (poi). And there're totally 21 features. Beside 'email_address' (string) and 'poi' (bool), all features are numeric.

It's a highly skewed data because the target 2 labels for classification are greatly unbalanced: poi people take up only 12.3% dataset.

Beside, the data has a lot of NA values of each features:


```python
persons = data_dict.keys()
keys = data_dict[data_dict.keys()[1]].keys()
```


```python
# Check 'NaN' value of each feature in dict_data
NA_count = {}

for person in persons:
    for key in keys:
        if data_dict[person][key] == 'NaN':
            if not NA_count.has_key(key):
                NA_count[key] = 1
            else:
                NA_count[key] += 1
```


```python
NA_count
```




    {'bonus': 64,
     'deferral_payments': 107,
     'deferred_income': 97,
     'director_fees': 129,
     'email_address': 35,
     'exercised_stock_options': 44,
     'expenses': 51,
     'from_messages': 60,
     'from_poi_to_this_person': 60,
     'from_this_person_to_poi': 60,
     'loan_advances': 142,
     'long_term_incentive': 80,
     'other': 53,
     'restricted_stock': 36,
     'restricted_stock_deferred': 128,
     'salary': 51,
     'shared_receipt_with_poi': 60,
     'to_messages': 60,
     'total_payments': 21,
     'total_stock_value': 20}



So the first thing needed to be done is to replace these 'NaN' values.
Because for financial data, 'NaN' most likely mean he/she had no such income, so it's 0 
And for email data, it can also be 0 if there's no such count, so I replace all numeric 'NaN' values by 0. Now most NA values are replaced, we only have people without email address, which is reasonable and can be ignored.


```python
# Before doing any feature selection and creation work, I remove the 'NaN'
# Because for financial data, 'NaN' most likely mean he/she had no such income, so it's 0
# And for email data, it can also be 0 if there's no such count
for person in persons:
    for key in keys:
        if data_dict[person][key] == 'NaN' and key != 'email_address':
            data_dict[person][key] = 0
```


```python
# Now NAs have been removed (Only email has NaN but it doesn't matter):
NA_count = {}

for person in persons:
    for key in keys:
        if data_dict[person][key] == 'NaN':
            if not NA_count.has_key(key):
                NA_count[key] = 1
            else:
                NA_count[key] += 1
NA_count
```




    {'email_address': 35}



The pros of such NA value replacement are:

- use reasonable logic to better fit the data
- good for training models

But there's con:
- some data might be real missing values (real value can be non-0!), for example, typo, unintended missing etc. Set them to 0 might have risk to bias the result

To go deeper, I have to assume the NA data be all values such person doesn't have.

Beside, outliers are harmful for analysis. After all, they can significantly biased any model. For example, if I use Decision Tree, outliers can setup new rules but actually it's sometimes meaningless. However, some outliers, if they are not a wiered point, and it belongs to the true data, I will keep them.

Here let's check if there's strange outliers. Let's pick up some feature combination and use scatter plot to check:



```python
### Task 3: Remove outliers

# Because I'm not using simple linear model for next steps 
# so if the outliers are from data points, even it's outlier, I will keep it
# otherwise I will remove it

# Firstly, let's do some visualization by using 2-dimensional data

features_list = ['deferral_payments','expenses']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)

import matplotlib.pyplot as plt

for point in data:
    deferral_payments = point[0]
    expenses = point[1]
    plt.scatter( deferral_payments, expenses )

plt.xlabel("deferral_payments")
plt.ylabel("expenses")
plt.title("Search for outliers")
plt.show()
```


![png](output_13_0.png)


We can see there's one point significantly far from center, let's see what it is:


```python
for person in persons:
    if data_dict[person]['deferral_payments'] > 20000000:
        print person
```

    TOTAL


It's unlikely for anyone to have name 'TOTAL', a more reasonable explaination is that this is summary of sum of feature values. So I remove this 'TOTAL' otherwise it might bias my models.


```python
data_dict.pop('TOTAL')
```




    {'bonus': 97343619,
     'deferral_payments': 32083396,
     'deferred_income': -27992891,
     'director_fees': 1398517,
     'email_address': 'NaN',
     'exercised_stock_options': 311764000,
     'expenses': 5235198,
     'from_messages': 0,
     'from_poi_to_this_person': 0,
     'from_this_person_to_poi': 0,
     'loan_advances': 83925000,
     'long_term_incentive': 48521928,
     'other': 42667589,
     'poi': False,
     'restricted_stock': 130322299,
     'restricted_stock_deferred': -7576788,
     'salary': 26704229,
     'shared_receipt_with_poi': 0,
     'to_messages': 0,
     'total_payments': 309886585,
     'total_stock_value': 434509511}



Then let's zoom in and see if there's other unreasonable outliers


```python
features_list = ['deferral_payments','expenses']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)

for point in data:
    deferral_payments = point[0]
    expenses = point[1]
    plt.scatter( deferral_payments, expenses )

plt.xlabel("deferral_payments")
plt.ylabel("expenses")
plt.xticks(range(-1000000, 8000000, 2000000))
plt.title("Search for outliers")
plt.show()
```


![png](output_19_0.png)


At least 4 outliers are detected visually. Let's look at these outliers


```python
persons.remove('TOTAL')
for person in persons:
    if data_dict[person]['deferral_payments'] > 5000000 or data_dict[person]['expenses'] > 150000:
        print person
```

    SHANKMAN JEFFREY A
    URQUHART JOHN A
    MCCLELLAN GEORGE
    FREVERT MARK A


These are all names of people, and they are not by mistakes. Since they include much information I need, to keep these outliers will be a good choice.

After check other combinations, I decide to keep all remaining data because the outliers will have a lot of information which could be significant indicator of poi.

## 2. Feature Selecting

The next task is to select feature I think most relevent to my research.
After thoroughly check all the feature meanings, I select:

- poi: the label I'm interested in
- deferral_payments: might lead to big fraud, because it's a kind of delayed payment in which the person could hide truth
- expenses: I suspect too much expenses might be an indicator of spend company's money on personal assets
- total_stock_value: certainly an indicator of wealth
- defered_income: same as deferral_payments
- total_payments: another indicator of wealth
- loan_advances: I don't think poi will have significant loans data, or might be I'm totally wrong
- long_term_incentive: this financial data is the most vague one here, if possible, I do hope to know how the incentive money be determined
- other: intuitively, I suspect poi might have big amount of money earned in blackbox

Beside, I create 3 features I think could possibly be indicators for poi:


- 'fixed_income': earned from how they contribute to work, salary + bonus
- 'stock_income': all income from stock, restricted_stock_deferred + exercised_stock_options + restricted_stock
- 'email_proportion_with_poi': 
- proportion of their emails frequency with poi over all email


```python
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','deferral_payments','expenses',
                 'total_stock_value','deferred_income',
                 'total_payments','loan_advances',
                 'long_term_incentive','other'] 
```


```python
### Task 3: Create new feature(s)
# Then I create features:
# created_feature_list:

for person in persons:
    salary = float(data_dict[person]['salary'])
    bonus = float(data_dict[person]['bonus'])
    restricted_stock_deferred = float(data_dict[person]['restricted_stock_deferred'])
    exercised_stock_options = float(data_dict[person]['exercised_stock_options'])
    restricted_stock = float(data_dict[person]['restricted_stock'])
    
    from_this_person_to_poi = float(data_dict[person]['from_this_person_to_poi'])
    shared_receipt_with_poi = float(data_dict[person]['shared_receipt_with_poi'])
    from_poi_to_this_person = float(data_dict[person]['from_poi_to_this_person'])
    to_messages = float(data_dict[person]['to_messages'])
    from_messages = float(data_dict[person]['from_messages'])
    
    data_dict[person]['fixed_income'] = salary + bonus 
    data_dict[person]['stock_income'] = (restricted_stock_deferred + \
                                         exercised_stock_options + \
                                         restricted_stock)
    data_dict[person]['email_proportion_with_poi'] = (from_this_person_to_poi + \
                                                         shared_receipt_with_poi + \
                                                         from_poi_to_this_person)/ \
                                                        (to_messages + from_messages + 1)
                                         
    

features_list = features_list + ['fixed_income', 'stock_income', 'email_proportion_with_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```

One more step before training algorithms is re-scale the data by sklearn.MinMaxScaler, because some of the algorithms I will implement might require re-scale features to avoid skewed distance and biased result.


```python
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)
```

Then I use sklearn KBest to select the best features


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

data_new_best = SelectKBest(f_classif).fit(features_minmax,labels)
```


```python
# fit_transform(X, y)
import operator

scores = data_new_best.scores_
score_dict = {}
for ii in range(11):
    score_dict[features_list[ii+1]] = round(scores[ii],2)

sorted_score_dict = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_score_dict
```




    [('total_stock_value', 24.47),
     ('fixed_income', 22.89),
     ('stock_income', 22.78),
     ('deferred_income', 11.6),
     ('email_proportion_with_poi', 10.26),
     ('long_term_incentive', 10.07),
     ('total_payments', 8.87),
     ('loan_advances', 7.24),
     ('expenses', 6.23),
     ('other', 4.2),
     ('deferral_payments', 0.22)]



I'm glad the 3 features I created are all listed in top 5. Now I can select the top 5 features and fit the data into the model.


```python
new_features_list = ['poi',
                     'total_stock_value', 
                     'fixed_income', 
                     'stock_income', 
                     'deferred_income', 
                     'email_proportion_with_poi']

my_dataset = data_dict

### Extract best features and labels from dataset
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```

## 3. Training and Tuning Machine Learning

Before training, I will use validation to see how algorithms generalized to the overall data outside training data. Without validation, there's pretty high risk of overfitting.

After that, I run each model independently:


```python
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

# classifier test function
def classifer_tester(classifier, features, labels, parameters, iterations=100):
    
    precision = []
    recall = []
    accuracy = []
    
    for ii in range(iterations):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=ii)
        grid_search = GridSearchCV(classifier, parameters)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        accuracy.append(accuracy_score(labels_test, predictions))
    
    precision_mean = np.array(precision).mean()
    recall_mean = np.array(recall).mean()
    accuracy_mean = np.array(accuracy).mean()
    
    print '------------------------'
    print 'Accuracy: %s' % "{:,.2f}".format(round(accuracy_mean, 2)) 
    print 'Precision: %s' % "{:,.2f}".format(round(precision_mean, 2))
    print 'Recall   : %s' % "{:,.2f}".format(round(recall_mean, 2))
    
    avg_F1 = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean)
    print 'F1 score:  %s' % "{:,.2f}".format(round(avg_F1, 2))
    
    print 'Best parameters:\n'
    best_parameters = grid_search.best_estimator_.get_params() 
    for parameter_name in sorted(parameters.keys()):
        print '%s: %r' % (parameter_name, best_parameters[parameter_name])

```

**(1) Naive Bayes**
As no parameters should be tuned, I will use defaulf for training.


```python
# Provided to give you a starting point. Try a variety of classifiers.

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
parameters = {}
grid_search = GridSearchCV(clf, parameters)
print '\nGaussian Naive Bayes:'
classifer_tester(clf, features, labels, parameters)
```

    
    Gaussian Naive Bayes:
    ------------------------
    Accuracy: 0.86
    Precision: 0.43
    Recall   : 0.37
    F1 score:  0.40
    Best parameters:
    



```python
## Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'],
               'min_samples_split': [2, 10, 20],
               'max_depth': [None, 2, 5, 10],
               'min_samples_leaf': [1, 5, 10],
               'max_leaf_nodes': [None, 5, 10, 20]}

grid_search = GridSearchCV(clf, parameters)
print '\nDecision Tree:'
classifer_tester(clf, features, labels, parameters)
```

    
    Decision Tree:
    ------------------------
    Accuracy: 0.84
    Precision: 0.16
    Recall   : 0.15
    F1 score:  0.16
    Best parameters:
    
    criterion: 'gini'
    max_depth: None
    max_leaf_nodes: None
    min_samples_leaf: 10
    min_samples_split: 2



```python
# ## Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# parameters = {'criterion': ['gini', 'entropy'],
#                'min_samples_split': [2, 10, 20],
#                'max_depth': [None, 2, 5, 10],
#                'min_samples_leaf': [1, 5, 10],
#                'max_leaf_nodes': [None, 5, 10, 20]}

# grid_search = GridSearchCV(clf, parameters)
# print '\nRandom Forest:'
# classifer_tester(clf, features, labels, parameters)
# This might take too long time, I just list the result here
```

Random Forest (Running the code to tune Random Forest takes too long time, so I just list the result here.)

Score Type| Score
----------|-------------|
Precision|0.30
Recall   | 0.15
F1 score | 0.20
Best parameters|None
criterion| 'gini'
max_depth| 2
max_leaf_nodes| None
min_samples_leaf| 1
min_samples_split| 10

The accuracy is an average score to show how much percentage we get the right prediction. As it's not suitable for skewed features, here I add precision and recall matrices to evaluate.

Precision is how much probability we get a sample with true if it's tested positive. By bayes prior and post probability:

$$Precision = P(T | +) = \frac{P(+|T)}{P(+|T)+P(+|F)}$$

In other words, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned:

$$Recall = \frac{P(+|T)}{P(+|T)+P(-|T)}$$


From above analysis, we can see Naive Bayes has better accuracy, recall, precision and f1 score. Naive Bayes algorithm is my best choice. 

## 4. Final Discuss with Naive Bayes


```python
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf = GaussianNB()
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
```

By Naive Beyes, the most significant indicators are:

I then run this model with test.py and get the result:

Accuracy: 0.85679	
Precision: 0.49840	
Recall: 0.38900	
F1: 0.43696	
F2: 0.40686
Total predictions: 14000	
True positives:  778	
False positives:  783	
False negatives: 1222	
True negatives: 11217

This is pretty close to the result I get with 146-length dataset.

The features I created also act as essential part of this model. Let's compare the result with / without the features I created.


```python
features_list = ['poi','deferred_income',
                 'total_stock_value'] 
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf = GaussianNB()
parameters = {}
grid_search = GridSearchCV(clf, parameters)
print '\nGaussian Naive Bayes:'
classifer_tester(clf, features, labels, parameters)
```

    
    Gaussian Naive Bayes:
    ------------------------
    Accuracy: 0.87
    Precision: 0.50
    Recall   : 0.34
    F1 score:  0.41
    Best parameters:
    


Score Type|With New Features|Without New Features
------------|------------|------------
Accuracy|0.86|0.87
Precision|0.43|0.50
Recall|0.37|0.34
F1 score|0.40|0.41

Even without the features I created, the model will have more accuracy, precision and f1 score, I prefer using with created features to keep recall rate higher. Since higher recall rate means lower risk to label non-poi if he's truely poi. We can set up inquiries beside this analysis to make further judgement, but missing a poi is harmful. So even sacrifice precision, I would keep the created new features to increase recall rate. 

## 5. Conclusion

In this report I firstly summarize the dataset, remove outliers and replace NaN values. Next, I create 3 features and figure out which features to be selected by sklearn KBest method. Then I rescale the dataset and use these features to train different model, and finally find Naive Bayes as my best model.

This is a quantative analysis and can only be a reference for commitment. The real procedure of convict guilty will be more complicated.

In future, to improve the accuracy of the model, I think there're some ways we can try:
- Given more detailed dataset, more features might have risk of overfitting, but more data can possibly provide more informaiton we need
- Mining more information from emails, for example, how they communicate with Enron's partners, how they poi communicate with each other


```python

```
