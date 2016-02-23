# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:39:15 2016

@author: DEGNINOU
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

"""
Data Management
"""
'''List of variables in the treeaddhealth.csv dataset:
'BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','AGE'	
'TREG1','ALCEVR1','ALCPROBS1','MAREVER1','COCEVER1','INHEVER1'	
'CIGAVAIL','DEP1','ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1'	
'GPA1','EXPEL1','FAMCONCT','PARACTV','PARPRES'
'''
#%%
data = pd.read_csv("tree_addhealth.csv")

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

# Data Management

data_clean = data.dropna()

# subset clustering variables
cluster=data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'AGE','ALCEVR1','MAREVER1','ALCPROBS1','DEVIANT1','VIOL1',
'DEP1','GPA1','SCHCONN1','PARACTV', 'PARPRES','FAMCONCT']]
cluster.describe()
#%%
# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['BIO_SEX']=preprocessing.scale(clustervar['BIO_SEX'].astype('float64'))
clustervar['HISPANIC']=preprocessing.scale(clustervar['HISPANIC'].astype('float64'))
clustervar['WHITE']=preprocessing.scale(clustervar['WHITE'].astype('float64'))
clustervar['BLACK']=preprocessing.scale(clustervar['BLACK'].astype('float64'))
clustervar['NAMERICAN']=preprocessing.scale(clustervar['NAMERICAN'].astype('float64'))
clustervar['ASIAN']=preprocessing.scale(clustervar['ASIAN'].astype('float64'))
clustervar['AGE']=preprocessing.scale(clustervar['AGE'].astype('float64'))
clustervar['ALCEVR1']=preprocessing.scale(clustervar['ALCEVR1'].astype('float64'))
clustervar['MAREVER1']=preprocessing.scale(clustervar['MAREVER1'].astype('float64'))
clustervar['ALCPROBS1']=preprocessing.scale(clustervar['ALCPROBS1'].astype('float64'))
clustervar['DEVIANT1']=preprocessing.scale(clustervar['DEVIANT1'].astype('float64'))
clustervar['VIOL1']=preprocessing.scale(clustervar['VIOL1'].astype('float64'))
clustervar['DEP1']=preprocessing.scale(clustervar['DEP1'].astype('float64'))
clustervar['GPA1']=preprocessing.scale(clustervar['GPA1'].astype('float64'))
clustervar['SCHCONN1']=preprocessing.scale(clustervar['SCHCONN1'].astype('float64'))
clustervar['PARACTV']=preprocessing.scale(clustervar['PARACTV'].astype('float64'))
clustervar['PARPRES']=preprocessing.scale(clustervar['PARPRES'].astype('float64'))
clustervar['FAMCONCT']=preprocessing.scale(clustervar['FAMCONCT'].astype('float64'))
#%%
# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-5 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,6)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
#%%
# Interpret 2 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()
#%%
"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']
#%%

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()
#%%
"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
#%%
# validate clusters in training data by examining cluster differences in ESTEEM using ANOVA
# first have to merge ESTEEM with clustering variables and cluster assignment data 
esteem_data=data_clean['ESTEEM1']
# split GPA data into train and test sets 
esteem_train, esteem_test = train_test_split(esteem_data, test_size=.3, random_state=123)
esteem_train1=pd.DataFrame(esteem_train)
esteem_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(esteem_train1, merged_train, on='index')
sub1 = merged_train_all[['ESTEEM1', 'cluster']].dropna()  

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

esteemod = smf.ols(formula='ESTEEM1 ~ C(cluster)', data=sub1).fit()
print (esteemod.summary())

print ('means for ESTEEM by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for ESTEEM by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['ESTEEM1'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
#%%
"""
VALIDATION
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster in test data set
"""
# create a variable out of the index for the cluster training dataframe to merge on
clus_test.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslistval=list(clus_test['index'])
# create a list of cluster assignments
labelsval=list(clusassignval)
# combine index variable list with labels list into a dictionary
newlistval=dict(zip(cluslistval, clusassignval))
newlistval
# convert newlist dictionary to a dataframe
newclusval=DataFrame.from_dict(newlistval, orient='index')
newclusval
# rename the cluster assignment column
newclusval.columns = ['cluster']
# create a variable out of the index for the cluster assignment dataframe to merge on
newclusval.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_test=pd.merge(clus_test, newclusval, on='index')
# cluster frequencies
merged_test.cluster.value_counts()
#%%
"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster 
"""

# calculate test data clustering variable means by cluster
clustergrpval = merged_test.groupby('cluster').mean()
print ("Test data clustering variable means by cluster")
print(clustergrpval)
#%%