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
print ('Summary statistics for Life Expectancy')
m1= gapmind['lifeexpectancy'].describe()
print (m1)

print ('Summary statistics for Policy Score')
m2= gapmind['polityscore'].describe()
print (m2)

print ('Summary statistics for Income per persone')
m3= gapmind['incomeperperson'].describe()
print (m3)

print ('Summary statistics for Female employement rate')
m4= gapmind['femaleemployrate'].describe()
print (m4)

print ('Summary statistics for Employement rate')
m5= gapmind['employrate'].describe()
print (m5)

print ('Summary statistics for Alcohol consumption')
m6= gapmind['alcconsumption'].describe()
print (m6)

print ('Summary statistics for CO2 Emissions')
m7= gapmind['co2emissions'].describe()
print (m7)

print ('Summary statistics for Residential electricity consumption')
m8= gapmind['relectricperperson'].describe()
print (m8)

print ('Summary statistics for Urbanicity')
m9= gapmind['urbanrate'].describe()
print (m9)

#%%
############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################
scat7 = seaborn.regplot(x="relectricperperson", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Residential electricity')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Residential electricity and Breast Cancers Rate')
print(scat7)

print ("OLS regression model for the Association Between Residential electricity and Breast Cancers Rate")
reg7 = smf.ols('breastcancer ~ relectricperperson', data=gapmind).fit()
print (reg7.summary())
#%%
scat7 = seaborn.regplot(x="urbanrate", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel("Urbanicity (%)")
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Urbanicity and Breast Cancers Rate')
print(scat7)

print ("OLS regression model for the Association Between Urbanicity and Breast Cancers Rate")
reg7 = smf.ols('breastcancer ~ urbanrate', data=gapmind).fit()
print (reg7.summary())

#%%
scat7 = seaborn.regplot(x="co2emissions", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('CO2 Emissions')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between CO2 Emissions and Breast Cancers Rate')
print(scat7)

print ("OLS regression model for the Association Between CO2 Emissions and Breast Cancers Rate")
reg7 = smf.ols('breastcancer ~ co2emissions', data=gapmind).fit()
print (reg7.summary())
#%%
scat6 = seaborn.regplot(x="alcconsumption", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Alcohol consumption')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Alcohol consumption and Breast Cancers Rate')
print(scat6)

print ("OLS regression model for the Association Between Alcohol consumption and Breast Cancers Rate")
reg6 = smf.ols('breastcancer ~ alcconsumption', data=gapmind).fit()
print (reg6.summary())
#%%
scat5 = seaborn.regplot(x="employrate", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Employement rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Employement and Breast Cancers Rate')
print(scat5)

print ("OLS regression model for the Association Between Employement and Breast Cancers Rate")
reg5 = smf.ols('breastcancer ~ employrate', data=gapmind).fit()
print (reg5.summary())
#%%
scat4 = seaborn.regplot(x="femaleemployrate", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Female employement rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Female employement and Breast Cancers Rate')
print(scat4)

print ("OLS regression model for the Association Between Female employement and Breast Cancers Rate")
reg4 = smf.ols('breastcancer ~ femaleemployrate', data=gapmind).fit()
print (reg4.summary())
#%%
scat3 = seaborn.regplot(x="polityscore", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Polity score')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Polity score and Breast Cancers Rate')
print(scat3)

print ("OLS regression model for the Association Between Polity score and Breast Cancers Rate")
reg3 = smf.ols('breastcancer ~ polityscore', data=gapmind).fit()
print (reg3.summary())
#%%
scat2 = seaborn.regplot(x="incomeperperson", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Income per person')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Income per persone and Breast Cancers Rate')
print(scat2)

print ("OLS regression model for the Association Between Income per person and Breast Cancers Rate")
reg2 = smf.ols('breastcancer ~ incomeperperson', data=gapmind).fit()
print (reg2.summary())
#%%
scat1 = seaborn.regplot(x="lifeexpectancy", y="breastcancer", scatter=True, data=gapmind)
plt.xlabel('Life Expectancy')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Life Expectancy and Breast Cancers Rate')
print(scat1)

print ("OLS regression model for the Association Between Life Expectancy and Breast Cancers Rate")
reg1 = smf.ols('breastcancer ~ lifeexpectancy', data=gapmind).fit()
print (reg1.summary())
#%%
#End of the code!
