# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 07:28:58 2016

@author: DEGNINOU
"""

#%%
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
gapmind['lifeexpectancy'] = pandas.to_numeric(gapmind['lifeexpectancy'], errors='coerce')
gapmind['breastcancer'] = pandas.to_numeric(gapmind['breastcancer'], errors='coerce')
gapmind['polityscore'] = pandas.to_numeric(gapmind['polityscore'], errors='coerce')
gapmind['incomeperperson'] = pandas.to_numeric(gapmind['incomeperperson'], errors='coerce')
gapmind['femaleemployrate'] = pandas.to_numeric(gapmind['femaleemployrate'], errors='coerce')
gapmind['employrate'] = pandas.to_numeric(gapmind['employrate'], errors='coerce')
gapmind['alcconsumption'] = pandas.to_numeric(gapmind['alcconsumption'], errors='coerce')
gapmind['co2emissions'] = pandas.to_numeric(gapmind['co2emissions'], errors='coerce')
gapmind['relectricperperson'] = pandas.to_numeric(gapmind['relectricperperson'], errors='coerce')
gapmind['urbanrate'] = pandas.to_numeric(gapmind['urbanrate'], errors='coerce')
#%%
# listwise deletion of missing values
gapmind1 = gapmind[['lifeexpectancy', 'breastcancer', 'polityscore', 'incomeperperson',
'femaleemployrate', 'employrate', 'alcconsumption', 'co2emissions', 'relectricperperson',
'urbanrate']].dropna() 
# Centering variables for regression 
gapmind1['lifeexpect_c'] = (gapmind1['lifeexpectancy'] - gapmind1['lifeexpectancy'].mean())
#gapmind1['breastcancer_c'] = (gapmind1['breastcancer'] - gapmind1['breastcancer'].mean())
gapmind1['income_c'] = (gapmind1['incomeperperson'] - gapmind1['incomeperperson'].mean())
gapmind1['femaleemplo_c'] = (gapmind1['femaleemployrate'] - gapmind1['femaleemployrate'].mean())
gapmind1['employrate_c'] = (gapmind1['employrate'] - gapmind1['employrate'].mean())
gapmind1['alcconsumption_c'] = (gapmind1['alcconsumption'] - gapmind1['alcconsumption'].mean())
gapmind1['co2emissions_c'] = (gapmind1['co2emissions'] - gapmind1['co2emissions'].mean())
gapmind1['relectric_c'] = (gapmind1['relectricperperson'] - gapmind1['relectricperperson'].mean())
gapmind1['urbanrate_c'] = (gapmind1['urbanrate'] - gapmind1['urbanrate'].mean())

#%%
############################################################################################
#LINEAR REGRESSION
############################################################################################
# linear regression analysis
print ("Regression model for the Association Between Urban Rate and Breast Cancers Rate")
reg1 = smf.ols('breastcancer ~ urbanrate_c', data=gapmind1).fit()
print (reg1.summary())

#%%
# quadratic (polynomial) regression analysis
print ('quadratic regression model for the Association Between Urban Rate and Breast Cancers Rate')
reg2 = smf.ols('breastcancer ~ urbanrate_c + I(urbanrate_c**2)', data=gapmind1).fit()
print (reg2.summary()) 
#%%
# first order (linear) scatterplot
scat1 = seaborn.regplot(x="urbanrate", y="breastcancer", scatter=True, data=gapmind1)
plt.xlabel('Urbanization rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Residential electricity and Breast Cancers Rate')
print(scat1)

# fit second order polynomial
scat2 = seaborn.regplot(x="urbanrate", y="breastcancer", scatter=True, order =2, data=gapmind1)
plt.xlabel('Urbanization rate')
plt.ylabel('Breast cancer new cases per 100,000 female')
plt.title ('Scatterplot for the Association Between Residential electricity and Breast Cancers Rate')
print(scat2)
#%%
############################################################################################
#MULTIPLE REGRESSION
############################################################################################
# Adding Life Expectancy
print ("Association Between Urban Rate, Life Expectancy and Breast Cancers Rate")
reg2 = smf.ols('breastcancer ~ urbanrate_c + lifeexpect_c', data=gapmind1).fit()
print (reg2.summary())
#%%
#Adding CO2 Emissions
print ("Association Between Urban Rate, Life Expectancy, CO2 Emissions and Breast Cancers Rate")
reg3 = smf.ols('breastcancer ~ urbanrate_c + lifeexpect_c + co2emissions_c', data=gapmind1).fit()
print (reg3.summary())
#%%
#Adding Income per person
print ("Association Between Urban Rate, Life Expectancy, Income, CO2 Emissions and Breast Cancers Rate")
reg4 = smf.ols('breastcancer ~ urbanrate_c + lifeexpect_c + co2emissions_c + income_c', data=gapmind1).fit()
print (reg4.summary())
#%%
#Adding alcohol consumption
print ("Association Between Urban Rate, Life Expectancy, Income, CO2 Emissions, Alcohol and Breast Cancers Rate")
reg5 = smf.ols('breastcancer ~ urbanrate_c + lifeexpect_c + co2emissions_c + income_c + alcconsumption_c', data=gapmind1).fit()
print (reg5.summary())
#%%
#Adding employement rate 
print ("Association Between Urban Rate, Life Expectancy, Income, CO2 Emissions, Alcohol, Employment and Breast Cancers Rate")
reg6 = smf.ols('breastcancer ~ urbanrate_c + lifeexpect_c + co2emissions_c + income_c + alcconsumption_c + employrate_c', data=gapmind1).fit()
print (reg6.summary())
#%%
#%%
#Keep only significant variables in the model 
print ("Association Between Income, Alcohol and Breast Cancers Rate")
reg7 = smf.ols('breastcancer ~ income_c + alcconsumption_c', data=gapmind1).fit()
print (reg7.summary())
####################################################################################
# EVALUATING MODEL FIT
####################################################################################
#%%
#Q-Q plot for normality
fig1=sm.qqplot(reg7.resid, line='r')
#%%
# simple plot of residuals
stdres=pandas.DataFrame(reg7.resid_pearson)
fig2 = plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print (fig2)
#%%
"""
# additional regression diagnostic plots
# For alcohol consumption 
fig3 = plt.figure(figsize=(12,8)) 
fig3 = sm.graphics.plot_regress_exog(reg7, 'alcconsumption_c', fig=fig3)
#%%
# For income 
fig4 = plt.figure(figsize=(12,8)) 
fig4 = sm.graphics.plot_regress_exog(reg7, 'income_c', fig=fig4)
"""
#%%
# leverage plot
fig5=sm.graphics.influence_plot(reg7, size=15)
print(fig5)
#%% End of the code ... 
