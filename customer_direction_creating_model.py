# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:54:18 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

dataset=pd.read_csv("NewAppData.csv")

#Data Preprocessing

response=dataset["enrolled"]
dataset=dataset.drop(columns="enrolled")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,response,test_size=0.2,random_state=0)


train_identifiers=X_train["user"]
X_train=X_train.drop(columns=["user"])
test_identifiers=X_test["user"]
X_test=X_test.drop(columns=["user"])


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_test2=pd.DataFrame(sc_X.transform(X_test))
X_train2.columns=X_train.columns.values
X_test2.columns=X_test.columns.values
X_train2.index=X_train.index.values
X_test2.index=X_test.index.values
X_train=X_train2
X_test=X_test2


#Building the model

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,scoring="accuracy",cv=10)
mean_accuracy=accuracies.mean()
deviation=accuracies.std()
print(f"Accuracy: {mean_accuracy} +- {deviation}")


final_result=pd.concat([y_test,test_identifiers],axis=1).dropna()
final_result["prediction"]=y_pred
final_result[["user","enrolled","prediction"]].reset_index(drop=True)

