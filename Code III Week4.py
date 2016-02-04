# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:32:51 2016

@author: DEGNINOU
"""

#%% Import libraries to be used 
import numpy 
import pandas 
import seaborn
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
gapmind = pandas.read_csv('gapminder.csv')

# convert variables to numeric format using convert_objects function
gapmind['breastcancer'] = pandas.to_numeric(gapmind['breastcancer'], 
errors='coerce')
gapmind['polityscore'] = pandas.to_numeric(gapmind['polityscore'], 
errors='coerce')
gapmind['incomeperperson'] = pandas.to_numeric(gapmind['incomeperperson'], 
errors='coerce')
gapmind['alcconsumption'] = pandas.to_numeric(gapmind['alcconsumption'], 
errors='coerce')
#%%
print ('Summary statistics for Breast Cancer Rate')
m1= gapmind['breastcancer'].describe()
print (m1)

print ('Summary statistics for Income per Person')
m2= gapmind['incomeperperson'].describe()
print (m2)

print ('Summary statistics for Alcohol Consumption')
m3= gapmind['alcconsumption'].describe()
print (m3)
#%%
#Creat binary Breast Cancer Rate
def bin2cancer (row):
   if row['breastcancer'] <= 20 :
      return 0
   elif row['breastcancer'] > 20 :
      return 1
#Apply the new variable bin2cancer to the gapmind dataset       
gapmind['bin2cancer'] = gapmind.apply (lambda row: bin2cancer (row),axis=1)
#Creat binary Income per person
def bin2income(row):
   if row['incomeperperson'] <= 5000 :
      return 0
   elif row['incomeperperson'] > 5000 :
      return 1
#Apply the new variable bin2income to the gapmind dataset  
gapmind['bin2income'] = gapmind.apply (lambda row: bin2income (row),axis=1)
#Creat binary Alcohol consumption
def bin2alcohol(row):
   if row['alcconsumption'] <= 5 :
      return 0
   elif row['alcconsumption'] > 5 :
      return 1
#Apply the new variable bin2alcohol to the gapmind dataset  
gapmind['bin2alcohol'] = gapmind.apply (lambda row: bin2alcohol (row),axis=1)
#%%
##############################################################################
#                    LOGISTIC REGRESSION 
##############################################################################
# logistic regression with binary income per persone
lreg1 = smf.logit(formula = 'bin2cancer ~ bin2income', 
                  data = gapmind).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
print ('Logistic regression with binary income per persone')
print ('Odd ratios with 95% confidence intervals')
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

# logistic regression with binary income per person 
# and binary alcohol consumption
lreg2 = smf.logit(formula = 'bin2cancer ~ bin2income + bin2alcohol', 
                  data = gapmind).fit()
print (lreg2.summary())

print ('Logistic regression with binary income per persone and binary alcohol consumption')
print ('Odd ratios with 95% confidence intervals')
# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
#%%
