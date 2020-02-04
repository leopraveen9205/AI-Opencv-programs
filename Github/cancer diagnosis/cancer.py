# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:48:15 2019

@author: rleo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("cancer diagnosis.csv")
data.columns
data.shape

data.dtypes

data.describe

data.drop(["id"],inplace=True,axis=1)
data.drop(["Bare Nuclei"],inplace=True,axis=1)
data.drop(["Unnamed"],inplace=True,axis=1)
data.describe()

desc=data.describe() ## describe mean, median, mode

sns.distplot(data["clump thick"])
sns.boxplot(data["clump thick"])

data["clump thick"]=data["clump thick"].astype("category")
data["diagnosis"]=data["diagnosis"].astype("category")

freq = pd.crosstab(data["diagnosis"],data["clump thick"])
from scipy.stats import chi2_contingency
print(chi2_contingency(freq))

clump_0 = data.loc[data["diagnosis"]==2,"clump thick"]
clump_1 = data.loc[data["diagnosis"]==4,"clump thick"]

from scipy.stats import ttest_ind
print(ttest_ind(clump_0,clump_1))


x = data["clump thick"]
y = data["diagnosis"]

# Importing the dataset
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

sns.distplot(data["u.cell size"])
sns.boxplot(data["u.cell size"])

sns.distplot(data["u.cell size"])
sns.boxplot(data["u.cell size"])

sns.distplot(data["Marg.Ad"])
sns.boxplot(data["Marg.Ad"])

sns.distplot(data["E.cell size"])
sns.boxplot(data["E.cell size"])

sns.distplot(data["Bland Chromatin"])
sns.boxplot(data["Bland Chromatin"])

sns.distplot(data["Normal Nuceli"])
sns.boxplot(data["Normal Nuceli"])

sns.distplot(data["mitoses"])
sns.boxplot(data["mitoses"])



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# coefficients of logisti regression 
print(classifier.coef_)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % cm)
accuracy=(sum(cm.diagonal())/cm.sum())*100
print("accuracy is :" ,accuracy)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Confusion matrix:\n%s" % acc)

