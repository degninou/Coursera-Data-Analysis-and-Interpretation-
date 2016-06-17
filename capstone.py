# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:42:25 2016

@author: DEGNINOU
"""
#%%
#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.pyplot as plp
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

#Load the dataset
data1 = pd.read_csv('worldbank.csv')
#keep variables of interest only
data1 = data1.dropna() #Clean dataset by droping lines with missing observations
data = data1[['country', 'x174_2013','x100_2013', 'x11_2013', 'x121_2013', 
'x125_2013' , 'x139_2013', 'x140_2013', 'x142_2013', 'x143_2013', 'x149_2013',  
'x14_2013', 'x150_2013', 'x155_2013', 'x156_2013', 'x157_2013', 'x169_2013', 
'x16_2013', 'x171_2013', 'x172_2013', 'x173_2013', 'x190_2013', 'x191_2013',  
'x192_2013', 'x1_2012', 'x204_2013', 'x205_2012', 'x67_2012', 'x68_2012', 
'x69_2012', 'x58_2013', 'x29_2013', 'x283_2013', 'x275_2013', 'x274_2013',
'x267_2013', 'x25_2013', 'x223_2013', 'x222_2013', 'x221_2013']]

#select predictor variables and target variable as separate data sets  
predvar = data[['x100_2013', 'x11_2013', 'x121_2013', 
'x139_2013', 'x140_2013', 'x143_2013', 'x149_2013', 'x14_2013', 
'x150_2013', 'x155_2013', 'x156_2013', 'x157_2013', 'x169_2013', 
'x16_2013', 'x171_2013', 'x172_2013', 'x173_2013', 
'x190_2013', 'x191_2013', 'x192_2013', 'x204_2013', 'x205_2012', 
'x67_2012', 'x68_2012', 'x69_2012', 'x29_2013', 'x283_2013', 
'x275_2013', 'x274_2013', 'x25_2013', 'x223_2013', 'x222_2013', 
'x221_2013', 'x142_2013', 'x58_2013', 'x1_2012', 
'x125_2013', 'x267_2013']]

target = data.x174_2013

#lable variables for graphing
dataplot = data.copy() 
dataplot['Risk_of_maternal_death']=dataplot['x174_2013']
dataplot['Death rate /1,000']=dataplot['x100_2013']
dataplot['Net national income /capita (US$)']=dataplot['x11_2013']
dataplot['Exports (% of GDP)']=dataplot['x121_2013']
dataplot['Total fertility rate']=dataplot['x125_2013'] 
dataplot['GDP (US$)']=dataplot['x139_2013']  
dataplot['GDP growth (annual %)']=dataplot['x140_2013']  
dataplot['GDP /capita (US$)']=dataplot['x142_2013']  
dataplot['GDP /capita growth (%)']=dataplot['x143_2013']  
dataplot['Health expend. /capita (US$)']=dataplot['x149_2013']  
dataplot['Adjusted savings(% of GNI)']=dataplot['x14_2013'] 
dataplot['Total health expend. (% of GDP)']=dataplot['x150_2013']  
dataplot['Improved sanitation (%)']=dataplot['x155_2013']  
dataplot['Improved water source (%)']=dataplot['x156_2013']  
dataplot['Incidence of TB /100,000']=dataplot['x157_2013']  
dataplot['Femal labor force (%)']=dataplot['x169_2013']  
dataplot['Adj. sav. educ. expend. (% of GNI)']=dataplot['x16_2013']  
dataplot['Female life expect. (years)']=dataplot['x171_2013']  
dataplot['Male life expect. (years)']=dataplot['x172_2013']  
dataplot['Total life expect. (years)']=dataplot['x173_2013']  
dataplot['Infant mortality rate /‰ LB']=dataplot['x190_2013']  
dataplot['Neonatal mortality rate /‰ LB']=dataplot['x191_2013']  
dataplot['Under-5 mortality rate /‰ LB']=dataplot['x192_2013']  
dataplot['Access to electric. (%)']=dataplot['x1_2012']  
dataplot['O-of-p health expend. (% of total)']=dataplot['x204_2013']  
dataplot['%female students primary educ. (%)']=dataplot['x205_2012']  
dataplot['C of death: mat., prenat., nutri. (%)']=dataplot['x67_2012']  
dataplot['C of death: injury (%)']=dataplot['x68_2012']  
dataplot['C of death: NCD (%)']=dataplot['x69_2012']  
dataplot['Crude birth rate /‰ people']=dataplot['x58_2013']  
dataplot['Age dependency ratio (%)']=dataplot['x29_2013'] 
dataplot['Urban population (%)']	=dataplot['x283_2013']  
dataplot['Male survival to age 65 (%)']=dataplot['x275_2013']  
dataplot['Female survival to age 65 (%)']=dataplot['x274_2013']  
dataplot['Strength rights index']=dataplot['x267_2013']  
dataplot['Ado. fertility rate']=dataplot['x25_2013'] 
dataplot['Female population (%)']=dataplot['x223_2013']  
dataplot['Population, 15-64 (%)']=dataplot['x222_2013']  
dataplot['Population, 0-14 (%)']=dataplot['x221_2013']

# quartile split (use qcut function & ask for 4 groups - gives you quartile split)
#GPD per capita - 4 categories - quartiles
dataplot['GPD_per_capita_2_cat']=pd.qcut(data1.x142_2013, 2, labels=["1=Low GDP/capita","2=High GDP/capita"])

# 2 split (use qcut function & ask for 3 groups - gives you quartile split)
#Birth rate, crude (per 1,000 people) - 3 categories
dataplot['Crude_birth_rate_2_cat']=pd.qcut(data1.x58_2013, 2, labels=["1=Low rate", "High rate"])

# 2 split (use qcut function & ask for 2 groups - gives you quartile split)
#Fertility rate, total (births per woman) - 2 categories
dataplot['Total_fertility_rate_2_cat']=pd.qcut(data1.x125_2013, 2, labels=["1=Low fertility","2=High fertility"])

# 2 split (use qcut function & ask for 2 groups - gives you quartile split)
#Strength of legal rights index - 2 categories
dataplot['Strength_of_rights_index_2_cat']=pd.qcut(data1.x267_2013, 2, labels=["1=Weak","2=Strong"])
#%%
#summary statistics of predictors and the target variable
predsumar = dataplot.describe() 
tarsummar = target.describe() 
predsumar.to_csv('predsumar.csv')
tarsummar.to_csv('tarsummar.csv')

#frequency distributions for categorical variables  
gdpcapita = pd.crosstab(index=dataplot["GPD_per_capita_2_cat"], columns="count")               # Name the count column
print ('Counts for GDP per capita')
print(gdpcapita) 
gdpcapita.to_csv('GDP per capita_count.csv')
#print (gdpcapita.sum(), "\n")   # Sum the counts
print ('Frequency % for GDP per capita')
gdpcapitafr = gdpcapita/gdpcapita.sum()
print (gdpcapitafr) 
gdpcapitafr.to_csv('GDP per capita_frequency.csv')

brate = pd.crosstab(index=dataplot["Crude_birth_rate_2_cat"], columns="count")               # Name the count column
print ('Counts for Crude birth rate')
print(brate) 
brate.to_csv('Crude birth rate_count.csv')
#print (gdpcapita.sum(), "\n")   # Sum the counts
print ('Frequency % for Crude birth rate')
bratefr = brate/brate.sum() 
print (bratefr)
bratefr.to_csv('Crude birth rate_frequency.csv')

fertil = pd.crosstab(index=dataplot["Total_fertility_rate_2_cat"], columns="count")               # Name the count column
print ('Counts for fertility rate')
print(fertil) 
fertil.to_csv('Fertility rate_count.csv')
#print (gdpcapita.sum(), "\n")   # Sum the counts
print ('Frequency % for fertility rate')
fertilfr = fertil/fertil.sum()
print (fertilfr)
fertilfr.to_csv('Fertility rate_frequency.csv')

legal = pd.crosstab(index=dataplot["Strength_of_rights_index_2_cat"], columns="count")               # Name the count column
print ('Counts for legal rights')
print(legal) 
legal.to_csv('Legal rights_count.csv')
#print (gdpcapita.sum(), "\n")   # Sum the counts
print ('Frequency % for legal rights')
legalfr = legal/legal.sum()
print (legalfr)
legalfr.to_csv('Legal rights_frequency.csv')
#%%
# split data for scatterplots
train,test=train_test_split(dataplot, test_size=.4, random_state=123)
#%
#scatterplot matrix for quantitative variables
fig1 = sns.PairGrid(train, y_vars=["Risk_of_maternal_death"], 
                 x_vars=['Death rate /1,000',\
                 'Population, 0-14 (%)',\
                 'Population, 15-64 (%)',\
                 'Exports (% of GDP)'], palette = 'GnBu_d')
fig1.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig1 = plt.gcf() 
fig1.set_size_inches(15, 8)
fig1.savefig('reportfig1.jpg')
#
#scatterplot matrix for quantitative variables -continued
fig2 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'],
                    x_vars=['GDP (US$)', 'GDP growth (annual %)',\
                    'Age dependency ratio (%)',\
                    'GDP /capita growth (%)',\
                    'Net national income /capita (US$)'], palette = 'GnBu_d')
fig2.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig2 = plt.gcf()
fig2.set_size_inches(15, 8)
fig2.savefig('reportfig2.jpg')
#
#scatterplot matrix for quantitative variables -continued
fig3 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['Health expend. /capita (US$)',\
                 'Adjusted savings(% of GNI)',\
                 'Total health expend. (% of GDP)',\
                 'Improved sanitation (%)'],\
                 palette = 'GnBu_d')
fig3.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig3 = plt.gcf()
fig3.set_size_inches(15, 8)
fig3.savefig('reportfig3.jpg')
#
#scatterplot matrix for quantitative variables -continued
fig4 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['Improved water source (%)',\
                 'Incidence of TB /100,000',\
                 'Femal labor force (%)',\
                 'Adj. sav. educ. expend. (% of GNI)'],\
                 palette = 'GnBu_d')
fig4.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig4 = plt.gcf()
fig4.set_size_inches(15, 8)
fig4.savefig('reportfig4.jpg')
#%
#scatterplot matrix for quantitative variables -continued
fig5 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['Female life expect. (years)',\
                 'Male life expect. (years)',\
                 'Total life expect. (years)',\
                 'Infant mortality rate /‰ LB'],\
                 palette = 'GnBu_d')
fig5.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig5 = plt.gcf()
fig5.set_size_inches(15, 8)
fig5.savefig('reportfig5.jpg')
#
#scatterplot matrix for quantitative variables -continued
fig6 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['Neonatal mortality rate /‰ LB',\
                 'Under-5 mortality rate /‰ LB',\
                 'O-of-p health expend. (% of total)',\
                 'Urban population (%)'],\
                 palette = 'GnBu_d')
fig6.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig6 = plt.gcf()
fig6.set_size_inches(15, 8)
fig6.savefig('reportfig6.jpg')
#%
#scatterplot matrix for quantitative variables -continued
fig7 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['%female students primary educ. (%)',\
                 'C of death: mat., prenat., nutri. (%)',\
                 'C of death: injury (%)',\
                 'C of death: NCD (%)'],\
                 palette = 'GnBu_d')
fig7.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig7 = plt.gcf()
fig7.set_size_inches(15, 8)
fig7.savefig('reportfig7.jpg')
#%
#scatterplot matrix for quantitative variables -continued
fig8 = sns.PairGrid(train, y_vars=['Risk_of_maternal_death'], 
                 x_vars=['Female survival to age 65 (%)',\
                 'Ado. fertility rate',\
                 'Female population (%)',\
                 'Male survival to age 65 (%)'],\
                 palette = 'GnBu_d')
fig8.map(plt.scatter, s=80, edgecolor="red")
#plt.title('Figure 1. Association Between WDI and Risk of maternal death (%)', 
#fontsize = 12, loc='right')
fig8 = plt.gcf()
fig8.set_size_inches(15, 8)
fig8.savefig('reportfig8.jpg')
#%%
# boxplots for association between categorical predictors & response
box1 = sns.boxplot(x="Crude_birth_rate_2_cat", y="Risk_of_maternal_death", data=train)
box1 = plt.gcf()
box1.savefig('reportfig9.jpg')

# using ols function for calculating the F-statistic and associated p value
print ('ANOVA comparision of maternal death risk across Birth rate categories')
model1 = smf.ols(formula='Risk_of_maternal_death ~ C(Crude_birth_rate_2_cat)', data=train)
results1 = model1.fit()
print (results1.summary())
#%%
# boxplots for association between categorical predictors & response
box2 = sns.boxplot(x="Total_fertility_rate_2_cat", y="Risk_of_maternal_death", data=train)
box2 = plt.gcf()
box2.savefig('reportfig10.jpg')

# using ols function for calculating the F-statistic and associated p value
print ('ANOVA comparision of maternal death risk across Fertility rate categories')
model2 = smf.ols(formula='Risk_of_maternal_death ~ C(Total_fertility_rate_2_cat)', data=train)
results2 = model2.fit()
print (results2.summary())
#%%
# boxplots for association between categorical predictors & response
box3 = sns.boxplot(x="Strength_of_rights_index_2_cat", y="Risk_of_maternal_death", data=train)
box3 = plt.gcf()
box3.savefig('reportfig11.jpg')

# using ols function for calculating the F-statistic and associated p value
print ('ANOVA comparision of maternal death risk across Legal rights index categories')
model3 = smf.ols(formula='Risk_of_maternal_death ~ C(Strength_of_rights_index_2_cat)', data=train)
results3 = model3.fit()
print (results3.summary())
#%%
# boxplots for association between categorical predictors & response
box4 = sns.boxplot(x="GPD_per_capita_2_cat", y="Risk_of_maternal_death", data=train)
box4 = plt.gcf()
box4.savefig('reportfig12.jpg')

# using ols function for calculating the F-statistic and associated p value
print ('ANOVA comparision of maternal death risk across GDP per capita categories')
model4 = smf.ols(formula='Risk_of_maternal_death ~ C(GPD_per_capita_2_cat)', data=train)
results4 = model4.fit()
print (results4.summary())
#%% 
# standardize predictors to have mean=0 and sd=1 for lasso regression
predictors=predvar.copy()
from sklearn import preprocessing
predictors['x100_2013']=preprocessing.scale(predictors['x100_2013'].astype('float64'))
predictors['x11_2013']=preprocessing.scale(predictors['x11_2013'].astype('float64'))
predictors['x121_2013']=preprocessing.scale(predictors['x121_2013'].astype('float64'))
predictors['x139_2013']=preprocessing.scale(predictors['x139_2013'].astype('float64'))
predictors['x140_2013']=preprocessing.scale(predictors['x140_2013'].astype('float64'))
predictors['x143_2013']=preprocessing.scale(predictors['x143_2013'].astype('float64'))
predictors['x149_2013']=preprocessing.scale(predictors['x149_2013'].astype('float64'))
predictors['x14_2013']=preprocessing.scale(predictors['x14_2013'].astype('float64'))
predictors['x150_2013']=preprocessing.scale(predictors['x150_2013'].astype('float64'))
predictors['x155_2013']=preprocessing.scale(predictors['x155_2013'].astype('float64'))
predictors['x156_2013']=preprocessing.scale(predictors['x156_2013'].astype('float64'))
predictors['x157_2013']=preprocessing.scale(predictors['x157_2013'].astype('float64'))
predictors['x169_2013']=preprocessing.scale(predictors['x169_2013'].astype('float64'))
predictors['x16_2013']=preprocessing.scale(predictors['x16_2013'].astype('float64'))
predictors['x171_2013']=preprocessing.scale(predictors['x171_2013'].astype('float64'))
predictors['x172_2013']=preprocessing.scale(predictors['x172_2013'].astype('float64'))
predictors['x173_2013']=preprocessing.scale(predictors['x173_2013'].astype('float64'))
predictors['x190_2013']=preprocessing.scale(predictors['x190_2013'].astype('float64'))
predictors['x191_2013']=preprocessing.scale(predictors['x191_2013'].astype('float64'))
predictors['x192_2013']=preprocessing.scale(predictors['x192_2013'].astype('float64'))
predictors['x204_2013']=preprocessing.scale(predictors['x204_2013'].astype('float64'))
predictors['x205_2012']=preprocessing.scale(predictors['x205_2012'].astype('float64'))
predictors['x67_2012']=preprocessing.scale(predictors['x67_2012'].astype('float64'))
predictors['x68_2012']=preprocessing.scale(predictors['x68_2012'].astype('float64'))
predictors['x69_2012']=preprocessing.scale(predictors['x69_2012'].astype('float64'))
predictors['x29_2013']=preprocessing.scale(predictors['x29_2013'].astype('float64'))
predictors['x283_2013']=preprocessing.scale(predictors['x283_2013'].astype('float64'))
predictors['x275_2013']=preprocessing.scale(predictors['x275_2013'].astype('float64'))
predictors['x274_2013']=preprocessing.scale(predictors['x274_2013'].astype('float64'))
predictors['x25_2013']=preprocessing.scale(predictors['x25_2013'].astype('float64'))
predictors['x223_2013']=preprocessing.scale(predictors['x223_2013'].astype('float64'))
predictors['x222_2013']=preprocessing.scale(predictors['x222_2013'].astype('float64'))
predictors['x221_2013']=preprocessing.scale(predictors['x221_2013'].astype('float64'))
predictors['x142_2013']=preprocessing.scale(predictors['x142_2013'].astype('float64'))
predictors['x58_2013']=preprocessing.scale(predictors['x58_2013'].astype('float64'))
predictors['x1_2012']=preprocessing.scale(predictors['x1_2012'].astype('float64'))
predictors['x125_2013']=preprocessing.scale(predictors['x125_2013'].astype('float64'))
predictors['x267_2013']=preprocessing.scale(predictors['x267_2013'].astype('float64'))
#%
# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=.4, random_state=123)
#%
# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
#%
# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))
#regcoef.to_csv('variable+regresscoef.csv')
#%%
# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
#%
# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
#%       
# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)
#%
# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
#%%
