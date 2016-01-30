# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:12:16 2016

@author: DEGNINOU

"""
#%%
import numpy 
import pandas 
import seaborn
import matplotlib.pyplot as plt 
import statsmodels.api
import statsmodels.formula.api as smf

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
gapmind = pandas.read_csv('gapminder.csv')

# convert variables to numeric format using convert_objects function
gapmind['lifeexpectancy'] = pandas.to_numeric(gapmind['lifeexpectancy'], errors='coerce')
gapmind["breastcancer"] = pandas.to_numeric(gapmind["breastcancer"], errors='coerce')
gapmind['polityscore'] = pandas.to_numeric(gapmind['polityscore'], errors='coerce')
gapmind['incomeperperson'] = pandas.to_numeric(gapmind['incomeperperson'], errors='coerce')
gapmind['femaleemployrate'] = pandas.to_numeric(gapmind['femaleemployrate'], errors='coerce')
gapmind['employrate'] = pandas.to_numeric(gapmind['employrate'], errors='coerce')
gapmind['alcconsumption'] = pandas.to_numeric(gapmind['alcconsumption'], errors='coerce')
gapmind['co2emissions'] = pandas.to_numeric(gapmind['co2emissions'], errors='coerce')
gapmind['relectricperperson'] = pandas.to_numeric(gapmind['relectricperperson'], errors='coerce')
gapmind['urbanrate'] = pandas.to_numeric(gapmind['urbanrate'], errors='coerce')
#%%
print ('Summary statistics for Urbanicity')
m1= gapmind['urbanrate'].describe()
print (m1)
# mean (urban rate) = 56.77
#%%
#Centering urban rate variable
gapmind['curbanrate'] = numpy.nansum([gapmind['urbanrate'], -56.77], axis=0)

#%%
print ('Summary statistics for Centered Urban Rate')
m2= gapmind['curbanrate'].describe()
print (m2)
############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################

#%%
scat1 = seaborn.regplot(x="urbanrate", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Urban rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Urban rate and Breast Cancers Rate')
print(scat1)

print ("Regression model for the Association Between Urban rate and Breast Cancers Rate")
reg1 = smf.ols('breastcancer ~ urbanrate', data=gapmind).fit()
print (reg1.summary())
#%%
#%%
scat2 = seaborn.regplot(x="curbanrate", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Centered Urban Rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Centered Urban Rate and Breast Cancers Rate')
print(scat2)

print ("Regression model for Centered Urban Rate and Breast Cancers Rate")
reg2 = smf.ols('breastcancer ~ curbanrate', data=gapmind).fit()
print (reg2.summary())
#%%
#End of the code!
