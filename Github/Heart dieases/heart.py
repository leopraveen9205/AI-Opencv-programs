# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:03:41 2019

@author: rleo
"""

import pandas as pd
import seaborn as sns

data = pd.read_csv("heart.csv")

data.shape
data.columns
data.dtypes

## checking null values
data.isnull().any(axis=1)
#data.fillna("vlue to be given ")
#data["colm"].fillna("vlue to be given ")

for i in data.columns:
    print(i)
    print(data[i].unique())
    
## univariant analysis
sns.distplot(data["age"])
sns.boxplot(data["age"])

sns.distplot(data["sex"])
sns.boxplot(data["sex"])
data["sex"] = data["sex"].astype('continuous')

sns.distplot(data["cp"])
sns.boxplot(data["cp"])

sns.distplot(data["trestbps"])
sns.boxplot(data["trestbps"])

sns.distplot(data["chol"])
sns.boxplot(data["chol"])

sns.distplot(data["fbs"])
sns.boxplot(data["fbs"])

sns.distplot(data["restecg"])
sns.boxplot(data["restecg"])

sns.distplot(data["thalach"])
sns.boxplot(data["thalach"])

sns.distplot(data["exang"])
sns.boxplot(data["exang"])

sns.distplot(data["oldpeak"])
sns.boxplot(data["oldpeak"])

sns.distplot(data["slope"])
sns.boxplot(data["slope"])

sns.distplot(data["ca"])
sns.boxplot(data["ca"])

sns.distplot(data["thal"])
sns.boxplot(data["thal"])

sns.distplot(data["target"])
sns.boxplot(data["target"])

data.drop(["fbs","ca"],inplace=True,axis=1)

from scipy.stats import ttest_ind

for i in data.columns:
    if i!="target":
        print(i)
        grp_0 = data.loc[data["target"]==0,i]
        grp_1 = data.loc[data["target"]==1,i]
        print(ttest_ind(grp_0,grp_1))
        
sec3=data.describe()

x = data[['age', 'sex', 'cp', 'trestbps', 'chol','restecg', 'thalach', 'exang','oldpeak','slope','thal']]
y = data["target"]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# coefficients of logisti regression 
print(classifier.coef_)
print(classifier.intercept_)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % cm)
accuracy=(sum(cm.diagonal())/cm.sum())*100
print("accuracy is :" ,accuracy)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="gini")
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)


from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier(criterion="entropy",n_estimators=10000)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)

from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier(criterion="gini",n_estimators=50)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)


#adaboost
from sklearn.ensemble import AdaBoostClassifier #For Classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

dt = DecisionTreeClassifier() 
lt=LogisticRegression()

classifier = AdaBoostClassifier(n_estimators=100, base_estimator=lt,learning_rate=1)
#classifier = AdaBoostClassifier(n_estimators=100, learning_rate=1)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)

#xgboost 
from xgboost.sklearn import XGBClassifier
nb = XGBClassifier(n_estimators=100)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy is :" ,accuracy*100)

# cross validation for over fitting
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = nb, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(estimator = nb,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print("Best accuracy is :/n",best_accuracy)
best_parameters = grid_search.best_params_
print("Best parameters are  :/n",best_parameters)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print("Best accuracy is :/n",best_accuracy)
best_parameters = grid_search.best_params_
print("Best parameters are  :/n",best_parameters)

