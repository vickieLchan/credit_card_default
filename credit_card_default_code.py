#!/usr/bin/env python
# coding: utf-8

# # Credit Card Default Predictor

# In[1]:


import csv
import pandas as pd
import numpy as np
import pickle
from sklearn_pandas import DataFrameMapper

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats

import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
get_ipython().magic('matplotlib inline')

from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report, auc
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.externals.six import StringIO 
from IPython.display import Image 
import pydot
import pydotplus
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from ipywidgets import interactive, FloatSlider

import psycopg2 as pg
from psycopg2 import connect
import pandas.io.sql as pd_sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from xgboost import XGBClassifier
from collections import Counter
from sqlalchemy import create_engine

from mlxtend.plotting import plot_decision_regions
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'png'")


# In[2]:


df = pd.read_csv('creditcard_defaults.csv')


# ## Data cleaning

# In[3]:


df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0))
df.columns = df.columns.str.lower()
df = (df.drop(columns = ['id'])).reset_index(drop = True)
df = df.astype(int)

# df.head()
# df.columns
# df.info
# df.dtypes
# sns.pairplot(df)


# ## Feature Engineering

# In[4]:


df['pay_timeliness'] = (df['pay_0'] + df['pay_2'] + df['pay_3'] + df['pay_4'] + df['pay_5'] + df['pay_6'])/6
df['bill_total'] = (df['bill_amt2'] + df['bill_amt3'] + df['bill_amt4'] + df['bill_amt5'] + df['bill_amt6'])/5
df['payment_total'] = (df['pay_amt1'] + df['pay_amt2'] + df['pay_amt3'] + df['pay_amt4'] + df['pay_amt5'])/5
df['percent_paid'] = df['payment_total'] / df['bill_total'] 
#We can add a new feature, an interaction term by creating a new column that takes payment total divided 
#by the bill total to get the percent of the total credit card bill that was paid over the past 6 months.
df = df.rename({'default payment next month': 'default'}, axis = 1)

#We can create multiple dummy variables for our categorical data including sex, education, and marriage.
df['marriage'].replace(0,3, inplace=True)
#turn all 0s into 3 for other. 

df['education'].replace({0:4, 5:4, 6:4}, inplace=True)

# df.head()

dummy_gender = pd.get_dummies(df['sex'])
dummy_education = pd.get_dummies(df['education'])
dummy_marriage = pd.get_dummies(df['marriage'])

dummy_gender.rename(columns={1:'male', 2:'female'}, inplace=True)
dummy_education.rename(columns={1:'gradschool', 2:'university', 3:'highschool', 4:'other_edu'}, inplace=True)
dummy_marriage.rename(columns={1:'married', 2:'single', 3:'other_marital'}, inplace=True)

# dummy_gender.head()
# dummy_education.head()
# df.loc[df.education == 0]
# dummy_marriage.head()
# df.loc[df.marriage == 0]

final_df = pd.concat([df, dummy_gender, dummy_education, dummy_marriage], axis=1)
final_df.head()


# The payment timeliness column is the average timeliness of the payments over the past 6 months. A -2 means that the client did not use their credit card, -1 means that the client paid on time, 0 means that the client did not carry a balance, +1 means that the client paid one month late, +2 means the client paid 2 months late...so on up to 9 months late.

# In[5]:


# final_df.to_csv(r'/Users/coffeeshoes/Desktop/Project_3_files/raw_df.csv', index = None, header=True)


# ## EDA

# In[6]:


Counter(final_df.default)
final_df['limit_bal'].max()
#Converted to USD is $32,166 or $1,000,000 NT
final_df.limit_bal.min()
#Converted to USD is $321 or $10,000 NT
final_df.age.max()
#79
final_df.age.min()
#21
final_df.bill_total.max()
#859874.4
final_df.bill_total.min() 
#-67431.4
#A negative credit card balance means that the client overpaid from previous billings.
final_df.payment_total.max()
final_df.payment_total.min()
final_df.percent_paid.max()
final_df.loc[final_df.payment_total >= 250000]

negative = final_df[(final_df['bill_total'] < 0)]
# negative
zero = final_df[(final_df['bill_total'] == 0)]
# zero.head()


# ### Comparing plots for those that do default and those that do not default in different limit ranges

# In[7]:


# mask1 = (X_train.limit_bal>=10000) & (X_train.limit_bal<=247500)
# mask2 = (X_train.limit_bal>=247501) & (X_train.limit_bal<=495000)
# mask3 = (X_train.limit_bal>=495001) &(X_train.limit_bal<=742500)
# mask4 = (X_train.limit_bal>=742501) &(X_train.limit_bal<=1000000)


# In[8]:


# np.mean(y_train[mask1]), np.mean(y_train[mask2]), np.mean(y_train[mask3]), np.mean(y_train[mask4])


# makes sense, people with higher balances are wealthier and less likely than non-wealthy people to default.

# In[9]:


# fig, ax = plt.subplots(3, 1, figsize=(10, 12))
# count0, bins_0, _ = ax[0].hist(X_train.loc[(y_train==0),'limit_bal'], bins=25, range=(0,1000000))
# count1, bins_1, _ = ax[1].hist(X_train.loc[(y_train==1),'limit_bal'], bins=25, range=(0,1000000))
# ax[2].plot((bins_0[:-1]+bins_0[1:])/2,count1/(count1 + count0));


# In[10]:


# fig, ax = plt.subplots(3, 1, figsize=(10, 12))
# count0, bins_0, _ = ax[0].hist(X_train.loc[(y_train==0),'age'], bins=25, range=(0,100))
# count1, bins_1, _ = ax[1].hist(X_train.loc[(y_train==1),'age'], bins=25, range=(0,100))
# ax[2].plot((bins_0[:-1]+bins_0[1:])/2,count1/(count1 + count0));


# In[11]:


# fig, ax = plt.subplots(3, 1, figsize=(10, 12))
# count0, bins_0, _ = ax[0].hist(X_train.loc[(y_train==0),'percent_paid'], bins=25, range=(0,1))
# count1, bins_1, _ = ax[1].hist(X_train.loc[(y_train==1),'percent_paid'], bins=25, range=(0,1))
# ax[2].plot((bins_0[:-1]+bins_0[1:])/2,count1/(count1 + count0));


# ### Replace all null, infinite values and normalizing data

# There are null values in the percent_paid column because some of the rows have a bill total of 0, and the payment total cannot be divided by 0. So if a client has a bill total of 0, we will assign a 1 to their percent paid, meaning that they paid 100% and are in good standing. 

# In[12]:


final_df['percent_paid'] = final_df['percent_paid'].replace([np.inf, -np.inf], np.nan)
final_df.fillna(1, inplace=True)

# final_df.isna().any()

# check_zero = final_df[(final_df['bill_total'] == 0)]
# check_zero

final_df = final_df.astype(float)
# final_df.head()

#Now we can standardize our numerical data.
normalized = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
              'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

final_df[normalized] = final_df[normalized].apply(lambda x: (x-np.mean(x))/np.std(x))

# final_df.head()


# ### Visualizing data

# In[13]:


plt.figure(figsize=(6,22))

plt.subplot(6,1,1)
plt.hist(final_df['limit_bal'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['limit_bal'][final_df['default'] == 1], bins=3, alpha = 0.7, label =             'default')
plt.ylabel('Distr of defaults')
plt.xlabel('Limit Balance')
plt.legend()

plt.subplot(6,1,2)
plt.hist(final_df['sex'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['sex'][final_df['default'] == 1], bins=3, alpha = 0.7, label = 'default')
plt.ylabel('Distr of defaults')
plt.xlabel('Sex')
plt.legend()

plt.subplot(6,1,3)
plt.hist(final_df['education'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['education'][final_df['default'] == 1], bins=3, alpha = 0.7, label = 'default')
plt.ylabel('Distr of defaults')
plt.xlabel('Education')
plt.legend()

plt.subplot(6,1,4)
plt.hist(final_df['marriage'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['marriage'][final_df['default'] == 1], bins=3, alpha = 0.7, label = 'default')
plt.ylabel('Distr of defaults')
plt.xlabel('Marriage')
plt.legend()

plt.subplot(6,1,5)
plt.hist(final_df['age'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['age'][final_df['default'] == 1], bins=3, alpha = 0.7, label = 'default')
plt.ylabel('Distribution of defaults')
plt.xlabel('Age')
plt.legend()

plt.subplot(6,1,6)
plt.hist(final_df['pay_timeliness'][final_df['default'] == 0], bins=3, alpha = 0.7, label = 'not default')
plt.hist(final_df['pay_timeliness'][final_df['default'] == 1], bins=3, alpha = 0.7, label = 'default')
plt.ylabel('Distr of defaults')
plt.xlabel('Payment Timeliness')
plt.legend()

plt.show()


# In[14]:


# plt.figure(figsize=(14,5))

# plt.subplot(1,4,1)
# plt.scatter(final_df['limit_bal'], final_df['default'])
# plt.ylabel('default payment next month')
# plt.xlabel('limit balance')

# plt.subplot(1,4,2)
# plt.scatter(final_df['sex'], final_df['default'])
# plt.ylabel('default payment next month')
# plt.xlabel('sex')

# plt.subplot(1,4,3)
# plt.scatter(final_df['education'], final_df['default'])
# plt.ylabel('default payment next month')
# plt.xlabel('education')

# plt.subplot(1,4,4)
# plt.scatter(final_df['marriage'], final_df['default'])
# plt.ylabel('default payment next month')
# plt.xlabel('marriage')
# plt.show()


# Create new dataframe with features selected from feature importance to see if it will help with minimizing overfitting on models below.

# In[15]:


short_df = df.filter(['pay_0','pay_2','pay_3', 'pay_4', 'pay_5', 'pay_6', 'default'], axis=1)
short_df.head()


# In[16]:


# short_df.to_csv(r'/Users/coffeeshoes/Desktop/Project_3_files/short_df.csv', index = None, header=True)


# ## Modeling

# In[17]:


y = final_df['default']
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, random_state=52, stratify=y)


# ### Dummy Classifier

# In[18]:


dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_y_pred = dummy.predict(X_test)
dummy_y_predproba = dummy.predict_proba(X_test)[:,1]
print('Test score: ', accuracy_score(y_test, dummy_y_pred))


# ### Instantiating model, confusion matrix, and scoring

# In[19]:


def model_scores(model, X, y):
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    y_predproba = model.predict_proba(X_test)[:,1]
    y_pred_train = model.predict(X_train)
    y_predproba_train = model.predict_proba(X_train)[:,1]

    print('Accuracy score for test set:', accuracy_score(y_test, y_pred))
    print('Precision score for test set:', precision_score(y_test, y_pred, pos_label=1))
    print('Recall score for test set:', recall_score(y_test, y_pred, pos_label=1))
    print('F1 score for test set:', f1_score(y_test, y_pred,  pos_label=1))
    print('\nTrain AUC score:', roc_auc_score(y_train, y_predproba_train))
    print('\nTest AUC score:', roc_auc_score(y_test, y_predproba)) 
    
def plot_confusion(model, X, y):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(cm, index=['not default', 
    'default'], columns=['predicted not default', 
    'predicted default'])
    print(confusion)
    
def plot_roc(model, X, y):
    # calculate the fpr and tpr for all thresholds of the classification
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('ROC Curve for Credit Card Default')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def model_viz(model, X, y):
    predictions = model_scores(model, X, y)
    plot_confusion(model, X, y)
    plot_roc(model, X, y)


# ### Gradient Boosting Classifier

# In[20]:


learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2,
        random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    print()
    
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 1, max_features=2, max_depth = 2, 
        random_state = 0)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report")
print(classification_report(y_test, predictions))

y_scores_gb = gb.decision_function(X_test)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))


# ## Logistic Regression

# ***Regular LR with AUC of 0.71, Recall of 0.22***

# In[21]:


lr = LogisticRegression(penalty='l2', C=0.25)
model_viz(lr, X, y)
y_predproba = lr.predict_proba(X_test)[:,1]
print('Log Loss is:', log_loss(y_test, y_predproba))


# ***Logistic Regression after oversampling minority***

# In[22]:


# X = final_df.drop('default', axis=1)
# y = final_df['default']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)

# X = pd.concat([X_train, y_train], axis=1)

# not_default = X[X.default==0]
# default = X[X.default==1]

# default_upsampled = resample(default,
#                           replace=True, # sample with replacement
#                           n_samples=len(not_default), # match number in majority class
#                           random_state=27) # reproducible results

# # combine majority and upsampled minority
# upsampled = pd.concat([not_default, default_upsampled])

# # check new class counts
# upsampled.default.value_counts()

# # trying logistic regression again with the balanced dataset
# y_train = upsampled.default
# X_train = upsampled.drop('default', axis=1)

# lr = LogisticRegression(penalty='l1', C=0.25)
# lr_cv_score = cross_val_score(lr, X_train, y_train, cv=10)
# upsampled = lr.fit(X_train, y_train)
# upsampled_pred = upsampled.predict(X_test)

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, upsampled_pred))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, upsampled_pred))
# print('\n')
# print("=== All AUC Scores ===")
# print(lr_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Logistic Regression: ", lr_cv_score.mean())


# ***Logistic Regression again after undersampling majority***

# In[23]:


# # downsample majority
# not_default_downsampled = resample(not_default,
#                                 replace = False, # sample without replacement
#                                 n_samples = len(default), # match minority n
#                                 random_state = 27) # reproducible results

# # combine minority and downsampled majority
# downsampled = pd.concat([not_default_downsampled, default])

# downsampled.default.value_counts()

# y_train = downsampled.default
# X_train = downsampled.drop('default', axis=1)

# lr = LogisticRegression(penalty='l1', C=0.25)
# lr_cv_score = cross_val_score(lr, X_train, y_train, cv=10)
# undersampled = lr.fit(X_train, y_train)
# undersampled_pred = undersampled.predict(X_test)

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, undersampled_pred))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, undersampled_pred))
# print('\n')
# print("=== All AUC Scores ===")
# print(lr_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Logistic Regression: ", lr_cv_score.mean())


# <font color=red>***Logistic Regression after SMOTE - Severe underfitting, AUC of 0.72, recall of 0.62***</font>

# In[24]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

smotelr = LogisticRegression(penalty='l2', C=0.0001).fit(X_train, y_train)
smotelr_cv_score = cross_val_score(smotelr, X_train, y_train, cv=10)
smotelr_pred = smotelr.predict(X_test)
smotelr_predproba = smotelr.predict_proba(X_test)[:,1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smotelr_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smotelr_pred))
print('\n')
print("=== All AUC Scores ===")
print(smotelr_cv_score)
print('\n')
print("=== Mean Train AUC Score ===")
print("Mean AUC Score - Logistic Regression: ", smotelr_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smotelr_predproba))


# ***Logistic Regression with important features, AUC is 0.70 and recall is 0.54***

# In[25]:


X = short_df.drop('default', axis=1)
y = short_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

smotelr = LogisticRegression(penalty='l2', C=0.25).fit(X_train, y_train)
smotelr_cv_score = cross_val_score(smotelr, X_train, y_train, cv=10)
smotelr_pred = smotelr.predict(X_test)
smotelr_predproba = smotelr.predict_proba(X_test)[:,1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smotelr_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smotelr_pred))
print('\n')
print("=== All AUC Scores ===")
print(smotelr_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Logistic Regression: ", smotelr_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smotelr_predproba))


# #### Logistic Regression Cross Validation

# Best Penalty: l1<br>
# Best C: 0.2562693193528691<br>
# After tuning the hyperparameters, the best AUC we can get with Logistic Regression is about 0.74.

# In[26]:


# clr = LogisticRegressionCV(Cs=100, penalty='l1', cv=10, solver='liblinear', multi_class='ovr', max_iter=100, 
#                           n_jobs=1, tol=0.0001, verbose=0)

# model_viz(clr, X, y)
# y_predproba = clr.predict_proba(X_test)[:,1]
# print('Log Loss for test set is:', log_loss(y_test, y_predproba))

# y_pred2proba = clr.predict_proba(X_test2)[:,1]
# print('Log Loss for 2nd test set is:', log_loss(y_test2, y_pred2proba))


# ***Tuning hyperparameter for Logistic Regression***

# In[27]:


# from scipy.stats import uniform
# logistic = LogisticRegression()
# penalty = ['l1', 'l2']
# C = uniform(loc=0, scale=4)
# hyperparameters = dict(C=C, penalty=penalty)
# clf = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
# best_model = clf.fit(X, y)
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])


# ## KNN

# ***Regular KNN has AUC of 0.64, recall score of .02***

# In[28]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

knn = KNeighborsClassifier(n_neighbors=78)
y = final_df.default
X = final_df.drop('default', axis=1)
model_viz(knn, X, y)


# #### KNN with Cross Validation

# ***overfitting, AUC of 0.64, recall of .02***

# In[29]:


# knn_cv = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=None, n_neighbors=78, p=2,
#                      weights='uniform')
# cv_scores = cross_val_score(knn_cv, X, y, cv=5)
# model_viz(knn_cv, X, y)


# ***Tuning hyperparameter for CV KNN***

# In[30]:


# k_range = list(range(1, 101))
# weight_options = ['uniform', 'distance']
# param_dist = dict(n_neighbors=k_range, weights=weight_options)
# rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)

# # fit
# rand.fit(X, y)

# # scores
# rand.cv_results_


# In[31]:


# print(rand.best_score_)
# print(rand.best_params_)
# print(rand.best_estimator_)


# ## Decision Trees 

# <font color= 'red'>***Regular Decision Tree with AUC of 0.76, recall of 0.47***</font>

# In[32]:


y = final_df.default
X = final_df.drop('default', axis=1)
dt = DecisionTreeClassifier(max_depth=7, min_samples_split=150, min_samples_leaf=35)
model_viz(dt, X, y)


# In[33]:


dt.fit(X, y)
importances = dt.feature_importances_
np.round(importances, 3)


# ***Creating Decision Tree image***

# In[34]:


# features = list(X)
# # features

# dot_data = StringIO()
# export_graphviz(dt, out_file=dot_data, feature_names=features, filled=True, rounded=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# graph.write_png("dt_predictors.png")


# ### Decision Trees Cross Validation

# ***Severe underfitting, AUC of 0.73, Accuracy of 0.79*** <br>

# In[35]:


# dt_cv = DecisionTreeClassifier(max_depth=20, max_features=26, min_samples_split=4, min_samples_leaf=10)
# dt_cv_scores = cross_val_score(dt_cv, X, y, cv=10)
# model_viz(dt_cv, X, y)


# ***Using important features, AUC is 0.74, recall is 0.36***

# In[36]:


X = short_df.drop('default', axis=1)
y = short_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

dt = DecisionTreeClassifier(max_depth=20, max_features=6, min_samples_split=7, min_samples_leaf=10)
model_viz(dt, X, y)


# ### Testing confusion matrix to make sure it's accurate

# In[37]:


# y_test.iloc[1]
# t = y_test.reset_index(drop=True)
# t.head()

# predictions = dt.predict_proba(X_test)[:,1]

# s = pd.Series(predictions)
# s.iloc[1]
# s.head()

# unconfused = pd.concat([t, s], axis=1, ignore_index=True)

# unconfused['actual'] = unconfused[0]
# unconfused['predicted'] = unconfused[1]

# unconfused = unconfused.drop([0,1], axis=1)
# unconfused.head(10)

# unconfused['predicted'] = np.where(unconfused['predicted']>0.5, 1, 0)
# unconfused.head()

# unconfused.groupby(["actual", "predicted"]).size()


# #### Tuning threshold for Decision Tree

# In[38]:


X = final_df.drop('default', axis=1)
y = final_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

dt = DecisionTreeClassifier(max_depth=20, max_features=6, min_samples_split=7, min_samples_leaf=10)
model_viz(dt, X, y)

def make_confusion_matrix(model, threshold=0.5):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    fraud_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(fraud_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['not default', 'default'],
           yticklabels=['not default', 'default']);
    plt.xlabel('predicted')
    plt.ylabel('actual')


# In[39]:


interactive(lambda threshold: make_confusion_matrix(dt, threshold), threshold=(0.0,1.0,0.02))


# ***Tuning hyperparameter for Decision Trees***

# In[40]:


# from scipy.stats import randint 

# param_dist = {"max_depth": [1, 200], 
#               "max_features": randint(1, 36), 
#               "min_samples_leaf": randint(1, 200),
#               "min_samples_split": randint(2, 200),
#               "criterion": ["gini", "entropy"]} 

# tree = DecisionTreeClassifier() 
# tree_cv = RandomizedSearchCV(tree, param_dist, cv = 10) 
# tree_cv.fit(X, y) 

# print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
# print("Best score is {}".format(tree_cv.best_score_)) 


# ***Running Decision Trees again after oversampling minority***

# In[41]:


# X = final_df.drop('default', axis=1)
# y = final_df['default']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)

# X = pd.concat([X_train, y_train], axis=1)

# not_default = X[X.default==0]
# default = X[X.default==1]

# default_upsampled = resample(default,
#                           replace=True, # sample with replacement
#                           n_samples=len(not_default), # match number in majority class
#                           random_state=27) # reproducible results

# # combine majority and upsampled minority
# upsampled = pd.concat([not_default, default_upsampled])

# # check new class counts
# upsampled.default.value_counts()

# # trying decision trees again with the balanced dataset
# y_train = upsampled.default
# X_train = upsampled.drop('default', axis=1)

# dt = DecisionTreeClassifier(max_depth=100, max_features=22, min_samples_split=91, min_samples_leaf=73)
# dt_cv_score = cross_val_score(dt, X_train, y_train, cv=10)
# upsampled = dt.fit(X_train, y_train)
# upsampled_pred = upsampled.predict(X_test)

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, upsampled_pred))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, upsampled_pred))
# print('\n')
# print("=== All AUC Scores ===")
# print(dt_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Decision Tree: ", dt_cv_score.mean())


# ***Running Decision Trees after undersampling majority***

# In[42]:


# # downsample majority
# not_default_downsampled = resample(not_default,
#                                 replace = False, # sample without replacement
#                                 n_samples = len(default), # match minority n
#                                 random_state = 27) # reproducible results

# # combine minority and downsampled majority
# downsampled = pd.concat([not_default_downsampled, default])

# downsampled.default.value_counts()

# y_train = downsampled.default
# X_train = downsampled.drop('default', axis=1)

# dt = DecisionTreeClassifier(max_depth=100, max_features=22, min_samples_split=91, min_samples_leaf=73)
# dt_cv_score = cross_val_score(dt, X_train, y_train, cv=10)
# undersampled = dt.fit(X_train, y_train)
# undersampled_pred = undersampled.predict(X_test)

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, undersampled_pred))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, undersampled_pred))
# print('\n')
# print("=== All AUC Scores ===")
# print(dt_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Decision Tree: ", dt_cv_score.mean())


# ***Decision Tree after SMOTE, severe overfitting, AUC of 0.75, recall of 0.51***

# In[43]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=20, max_features=36, min_samples_split=100, min_samples_leaf=100)
smotedt = dt.fit(X_train, y_train)
smotedt_cv_score = cross_val_score(smotedt, X_train, y_train, cv=10)
smotedt_pred = smotedt.predict(X_test)
smotedt_predproba = smotedt.predict_proba(X_test)[:,1]


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smotedt_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smotedt_pred))
print('\n')
print("=== All AUC Scores ===")
print(smotedt_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", smotedt_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smotedt_predproba))


# ***Using important features and SMOTE, now it's underfitting, AUC is 0.75, recall is 0.58***

# In[44]:


X = short_df.drop('default', axis=1)
y = short_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=20, max_features=6, min_samples_split=100, min_samples_leaf=100)
smotedt = dt.fit(X_train, y_train)
smotedt_cv_score = cross_val_score(smotedt, X_train, y_train, cv=10)
smotedt_pred = smotedt.predict(X_test)
smotedt_predproba = smotedt.predict_proba(X_test)[:,1]


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smotedt_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smotedt_pred))
print('\n')
print("=== All AUC Scores ===")
print(smotedt_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", smotedt_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smotedt_predproba))


# #### Tuning Threshold for Decision Trees

# In[45]:


# pred_proba_df = pd.DataFrame(model.predict_proba(x_test))
# threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
# for i in threshold_list:
#     print ('\n******** For i = {} ******'.format(i))
#     Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
#     test_accuracy = metrics.accuracy_score(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
#                                            Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
#     print('Our testing accuracy is {}'.format(test_accuracy))

#     print(confusion_matrix(Y_test.as_matrix().reshape(Y_test.as_matrix().size,1),
#                            Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1)))


# ## Random Forest

# ***Regular RF is overfitting, AUC of 0.72, recall of 0.33***

# In[46]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

rf = RandomForestClassifier()
# n_estimators=65, max_depth=13
model_viz(rf, X, y)


# ***RF - n_estimators hyperparameter tuning***

# In[47]:


# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# train_results = []
# test_results = []
# for estimator in n_estimators:
#    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
#    rf.fit(X_train, y_train)
#    train_pred = rf.predict(X_train)
#    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#    roc_auc = auc(false_positive_rate, true_positive_rate)
#    train_results.append(roc_auc)
#    y_pred = rf.predict(X_test)
#    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#    roc_auc = auc(false_positive_rate, true_positive_rate)
#    test_results.append(roc_auc)
# from matplotlib.legend_handler import HandlerLine2D
# line1, = plt.plot(n_estimators, train_results, color='blue', label= 'Train AUC')
# line2, = plt.plot(n_estimators, test_results, color='red', label= 'Test AUC')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel('n_estimators')
# plt.show()


# ***RF - max_depth hyperparameter tuning***

# In[48]:


# max_depths = np.linspace(1, 32, 32, endpoint=True)
# train_results = []
# test_results = []
# for max_depth in max_depths:
#    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
#    rf.fit(X_train, y_train)
#    train_pred = rf.predict(X_train)
#    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#    roc_auc = auc(false_positive_rate, true_positive_rate)
#    train_results.append(roc_auc)
#    y_pred = rf.predict(X_test)
#    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#    roc_auc = auc(false_positive_rate, true_positive_rate)
#    test_results.append(roc_auc)
# from matplotlib.legend_handler import HandlerLine2D
# line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
# line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel('Tree depth')
# plt.show()


# #### Random Forest Cross Validation

# <font color='red'>***AUC of 0.77, recall of 0.21***</font>

# In[49]:


rfc = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=2)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(X_test)
#from RandomizedSearchCV {'n_estimators': 10, 'max_features': 'sqrt', 'max_depth': 2}
# {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': 2}

model_viz(rfc, X, y)


# #### Tuning threshold for Random Forest

# In[50]:


interactive(lambda threshold: make_confusion_matrix(rfc, threshold), threshold=(0.0,1.0,0.02))


# In[51]:


rfc.fit(X, y)
importances = rfc.feature_importances_
np.round(importances, 3)


# In[52]:


df.columns


# ***Using important features Random Forest, AUC of 0.73 and recall of 0.25***

# In[53]:


# X = short_df.drop('default', axis=1)
# y = short_df['default']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)

# rfc = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=2)
# model_viz(rfc, X, y)


# ***Random Forest after SMOTE, AUC of 0.76 and recall of 0.51***

# In[54]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=2)
smoterf = rf.fit(X_train, y_train)
smoterf_cv_score = cross_val_score(smoterf, X_train, y_train, cv=10)
smoterf_pred = smoterf.predict(X_test)
smoterf_predproba = smoterf.predict_proba(X_test)[:,1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smoterf_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smoterf_pred))
print('\n')
print("=== All AUC Scores ===")
print(smoterf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", smoterf_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smoterf_predproba))


# ***SMOTE with important features, underfitting, AUC of 0.74, recall of 0.58***

# In[55]:


X = short_df.drop('default', axis=1)
y = short_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=2)
smoterf = rf.fit(X_train, y_train)
smoterf_cv_score = cross_val_score(smoterf, X_train, y_train, cv=10)
smoterf_pred = smoterf.predict(X_test)
smoterf_predproba = smoterf.predict_proba(X_test)[:,1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smoterf_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smoterf_pred))
print('\n')
print("=== All AUC Scores ===")
print(smoterf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", smoterf_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smoterf_predproba))


# ***Tuning hyperparameters for CV Random Forest***

# In[56]:


# # number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 2)]
# # number of features at every split
# max_features = ['auto', 'sqrt']

# # max depth
# max_depth = [int(x) for x in np.linspace(2, 200, num = 3)]
# max_depth.append(None)
# # create random grid
# random_grid = {
#  'n_estimators': n_estimators,
#  'max_features': max_features,
#  'max_depth': max_depth
#  }
# # Random search of parameters
# rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the model
# rfc_random.fit(X_train, y_train)
# # print results
# print(rfc_random.best_params_)


# ## Bernoulli Naive Bayes

# ***Regular Bernoulli with AUC of 0.76 and recall of 0.52***

# In[57]:


X = final_df.drop('default', axis=1)
y = final_df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

nb_B = BernoulliNB()
model_viz(nb_B, X, y)


# #### Bernoulli Cross Validation, slight underfitting, AUC of 0.76, recall of 0.52 

# In[58]:


# bnb = BernoulliNB()
# bnb_cv_score = cross_val_score(bnb, X, y, cv=10, scoring='roc_auc')
# bnb.fit(X_train, y_train)
# bnb_predict = bnb.predict(X_test)

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, bnb_predict))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, bnb_predict))
# print('\n')
# print("=== All AUC Scores ===")
# print(bnb_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Bernoulli Bayes: ", bnb_cv_score.mean())


# <font color=red>***Bernoulli after SMOTE, AUC of 0.76, recall of 0.53***</font>

# In[59]:


y = final_df.default
X = final_df.drop('default', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

bnb = BernoulliNB()
smotebnb = bnb.fit(X_train, y_train)
smotebnb_cv_score = cross_val_score(smotebnb, X_train, y_train, cv=10)
smotebnb_pred = smotebnb.predict(X_test)
smotebnb_predproba = smotebnb.predict_proba(X_test)[:,1]

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, smotebnb_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, smotebnb_pred))
print('\n')
print("=== All AUC Scores ===")
print(smotebnb_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", smotebnb_cv_score.mean())
print('\n')
print("=== Test AUC Score ===")
print('\nTest AUC score:', roc_auc_score(y_test, smotebnb_predproba))


# ## Support Vector Machine

# In[60]:


# svc_rbf = SVC(gamma=2, C=1)
# model_viz(svc_rbf, X, y)


# In[61]:


# svc =  SVC(kernel="linear", C=0.025)
# svc.fit(X_train, y_train)

