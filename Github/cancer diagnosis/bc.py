# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:41:19 2019

@author: rleo
"""

import pandas as pd
import seaborn as sns

data = pd.read_excel("bcdataset.xlsx")

data.shape
data.columns
data.dtypes

##
data.columns=['id', 'clump thick', 'u.cell size','Ucellshape','Marg.Ad',
       'E.cell size', 'Bare Nuclei','Bland Chromatin', 'Normal Nuceli', 'mitoses',
       'diagnosis', 'Unnamed: 11']



data.drop(["id","Unnamed: 11"],inplace=True,axis=1)

## checking null values
data.isnull().any(axis=1)
#data.fillna("vlue to be given ")
#data["colm"].fillna("vlue to be given ")

for i in data.columns:
    print(i)
    print(data[i].unique())
    
## select the particular coloumn and giving the value for it for missing value
    
data.loc[data["Bare Nuclei"]=='?',"Bare Nuclei"]=1

data["Bare Nuclei"]=data["Bare Nuclei"].astype(int)

data["diagnosis"].value_counts()
sns.countplot(data["diagnosis"])



## replace the value for daignosis
data.loc[data["diagnosis"]==2,"diagnosis"]=0
data.loc[data["diagnosis"]==4,"diagnosis"]=1

data.dtypes

data["diagnosis"] = data["diagnosis"].astype('category')

## univariant analysis
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

## skewness removal
data.loc[data["Marg.Ad"]>8,"Marg.Ad"]=8
data.loc[data["mitoses"]>3,"mitoses"]=3

from scipy.stats import ttest_ind

for i in data.columns:
    if i!="diagnosis":
        print(i)
        grp_0 = data.loc[data["diagnosis"]==0,i]
        grp_1 = data.loc[data["diagnosis"]==1,i]
        print(ttest_ind(grp_0,grp_1))        

x = data[['clump thick', 'u.cell size', 'Ucellshape', 'Marg.Ad', 'E.cell size',
       'Bare Nuclei', 'Bland Chromatin', 'Normal Nuceli']]
y = data["diagnosis"]

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

