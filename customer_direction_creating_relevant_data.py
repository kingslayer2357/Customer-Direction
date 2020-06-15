# -*- coding: utf-8 -*-
"""
Created on Mon May 18 03:16:04 2020

@author: kingslayer
"""

##### CUSTOMER APP SUBSCRIPTION DIRECTING #####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dateutil import parser

dataset=pd.read_csv(r"Appdata10.csv")

dataset.describe()

#Data cleaning

dataset["hour"]=dataset.hour.str.slice(1,3).astype(int)


#Plotting
dataset2=dataset.copy().drop(columns=["user","first_open","enrolled","screen_list","enrolled_date"])
dataset2.head()

#Histograms
plt.suptitle("Histograms of features",fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    
    vals=np.size(dataset2.iloc[:,i-1].unique())
    
    plt.hist(dataset2.iloc[:,i-1],bins=vals,color="green")
    plt.show()
    
    
#Correlation with response

dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),title="Co-Relation",grid=True,fontsize=45)


#Correlation matrix
sns.set(style="white",font_scale=2)

#compute the correlation
corr=dataset2.corr()

#set us matplot figure
f,ax=plt.subplots(figsize=(18,15))
f.suptitle("CORRELATION MATRIX",fontsize=40)

#Heatmap
sns.heatmap(corr,annot=True)




#Feature Engineering


dataset.dtypes
dataset["first_open"]=[parser.parse(row_data) for row_data in dataset["first_open"]]
dataset["enrolled_date"]=[parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in dataset["enrolled_date"]]
dataset.dtypes


dataset['difference']=(dataset.enrolled_date-dataset.first_open).astype("timedelta64[h]")

plt.hist(dataset["difference"])
plt.title("DIFFERNCE OF TIME SERIES")
plt.show()

plt.hist(dataset["difference"],range=[0,500])
plt.title("DIFFERNCE OF TIME SERIES")
plt.show()

plt.hist(dataset["difference"].dropna(),range=[0,100])
plt.title("DIFFERNCE OF TIME SERIES")
plt.show()


dataset.loc[dataset["difference"]>48,"enrolled"]=0

dataset=dataset.drop(columns=['first_open','difference','enrolled_date'])


#Feature engineering the screen list column

top_screens=pd.read_csv("Top_screens.csv").top_screens.values

dataset["screen_list"]=dataset.screen_list.astype(str)+","

for sc in top_screens:
    dataset[sc]=dataset["screen_list"].str.contains(sc).astype(int)
    dataset["screen_list"]=dataset.screen_list.str.replace(sc+",","")
    
dataset["others"]=dataset.screen_list.str.count(",")
dataset=dataset.drop(columns=["screen_list"])


#Funnels

saving_screens=["Saving1","Saving2","Saving2Amount","Saving4","Saving5","Saving6","Saving7","Saving8","Saving9","Saving10"]
dataset["SavingsCount"]=dataset[saving_screens].sum(axis=1)
dataset=dataset.drop(columns=saving_screens)

cm_screens=["Credit1","Credit2","Credit3","Credit3Container","Credit3Dashboard"]
dataset['CMCount']=dataset[cm_screens].sum(axis=1)
dataset=dataset.drop(columns=cm_screens)

cc_screens=["CC1","CC1Category","CC3"]
dataset["CCCount"]=dataset[cc_screens].sum(axis=1)
dataset=dataset.drop(columns=cc_screens)

loan_screens=["Loan","Loan2","Loan3","Loan4"]
dataset["LoanCount"]=dataset[loan_screens].sum(axis=1)
dataset=dataset.drop(columns=loan_screens)

dataset.to_csv("NewAppData.csv",index=False)


