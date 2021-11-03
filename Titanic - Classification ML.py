#!/usr/bin/env python
# coding: utf-8

# In[180]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[181]:


df = pd.read_csv("/Users/salmanosman/Downloads/titanic.csv")


# In[182]:


df


# In[183]:


df.isnull()


# In[184]:


df.isnull().sum()


# # Cleaning the Data

# In[185]:


df = df.dropna(subset = ['deck','embark_town'])


# In[186]:


df.isnull().sum()


# In[187]:


df.describe(include = 'all')


# In[188]:


df['age'] = df['age'].fillna(df['age'].mean())


# In[189]:


df.isnull().sum()


# In[190]:


df['age'] = df['age'].astype(int)


# In[191]:


df['age']


# In[192]:


df = df.drop(columns = ['alive','class'])


# The 'alive' columns is the same as 'survived'. If someone is not alive, he has not survived and vice-versa. Likewise, 'class' is a categorical representation of 'pclass', so it is not needed.

# In[193]:


df


# In[194]:


df = df.drop(columns = ['adult_male','embarked'])


# 'Adult_male' incorportated into 'who', so I have removed it as a column. 'Embarked' is a duplicate of 'embark_town', so it is also not needed.

# In[195]:


# Visually Examining data to see outliers


# In[196]:


sns.boxplot(x = df['pclass'], y = df['fare'])


# In[197]:


sns.boxplot(df['fare'])


# In[198]:


#Removing outliers in fare


# In[199]:


from scipy import stats
import numpy as np


# In[200]:


z_score = stats.zscore(df['fare'])
abs_z_score = np.abs(z_score)
filtered_fare = (abs_z_score < 3)


# In[201]:


df_1 = df[filtered_fare]


# In[202]:


df_1.describe(include = 'all')


# In[203]:


sns.boxplot(df_1['fare'])


# In[204]:


#Outliers successfully removed from fares


# In[205]:


#Further visualisations 


# In[206]:


sns.scatterplot(x=df_1['fare'], y=df_1['survived']);


# In[207]:


sns.scatterplot(x=df_1['age'], y=df_1['fare'], hue = df_1['survived'], size = df_1['pclass']);


# The scatter plot above shows that most of those that survived were adults in first class. 

# In[208]:


sns.scatterplot(x=df_1['age'], y=df_1['fare'], hue = df_1['pclass'], size = df_1['survived']);


# In[209]:


sns.violinplot(x=df_1['survived'],y=df_1['age'])


# This violin plot shows that very few people survived who were over 60, and bulk of those that survived were adults between 30 and 40 years of age.

# In[210]:


sns.regplot(x=df_1['age'],y=df_1['fare']);


# In[215]:


df_2 = df_1.groupby(['sex'], as_index = False)['survived'].sum()
df_2


# In[216]:


df_3 = df_1.groupby(['who'], as_index = False)['survived'].sum()
df_3


# In[261]:


df_4 = df_1.groupby(['pclass'], as_index = False)['survived'].sum()
df_4


# In[140]:


#Correlations


# In[217]:


df_1.corr()


# In[218]:


#No apparent relationship between survival and any of numeric independent variables 


# #Determining p-values

# In[222]:


# Survived vs age 


# In[223]:


pearson_coef, p_value = stats.pearsonr(df_1['age'], df_1['survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[224]:


# Weak relationship between age and survival, however p-value suggests that it is significant.


# In[225]:


# Survived vs sibsp


# In[226]:


pearson_coef, p_value = stats.pearsonr(df_1['sibsp'], df_1['survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[227]:


# Weak relationship between sibsp and survived and moderate evidence that the correlation is significant


# In[228]:


# Survived vs parch


# In[229]:


pearson_coef, p_value = stats.pearsonr(df_1['parch'], df_1['survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[230]:


# Weak relationship between parch and survived. Correlation is insignificant as p-value is > 0.1


# In[234]:


# Survived vs alone


# In[235]:


pearson_coef, p_value = stats.pearsonr(df_1['alone'], df_1['survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[236]:


# Weak relationship between alone and survived, moderate evidence that the correlation is significant.


# #Of numerical variables, include pclass, fare, Age, Sibsp and alone in model. Discard others.

# In[263]:


df_1


# In[239]:


df_1 = df_1.drop(columns = ['embark_town'])


# In[240]:


df_1


# In[241]:


sex_dummies = pd.get_dummies(df_1['sex'], prefix = 'sex')
who_dummies = pd.get_dummies(df['who'], prefix = 'who')
deck_dummies = pd.get_dummies(df['deck'], prefix = 'deck')
alone_dummies = pd.get_dummies(df['alone'], prefix = 'alone')


# In[242]:


df_1 = pd.concat([df_1, sex_dummies, who_dummies, deck_dummies, alone_dummies], axis =1)


# In[247]:


df_1


# In[248]:


df_1 = df_1.drop(columns = ['deck'])


# In[249]:


df_1


# In[250]:


df_1 = df_1.dropna()


# In[251]:


df_1['sex_female'] = df_1['sex_female'].astype(int)


# In[252]:


df_1['sex_male'] = df_1['sex_male'].astype(int)


# In[254]:


df_1


# In[255]:


df_1.corr()


# 

# Remove deck from the model, it appears to have no correlation to 'survived', and also initially it had a lot of missing data.

# In[256]:


df_1 = df_1.drop(columns = ['deck_A','deck_B','deck_C','deck_D','deck_E','deck_F','deck_G'])


# In[262]:


df_1


# In[258]:


pearson_coef, p_value = stats.pearsonr(df_1['parch'], df_1['survived'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# 'parch' has a P-value of 0.6, so it is insignificant and should be removed.

# In[264]:


df_1


# In[265]:


df_1 = df_1.drop(columns = ['sex','who','alone'])


# In[269]:


df_1


# In[284]:


pclass_dummies = pd.get_dummies(df_1['pclass'], prefix = 'pclass')
pclass_dummies.sample(n=5)


# In[288]:


df_1 = pd.concat([df_1, pclass_dummies], axis =1)
df_1


# In[282]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


# In[268]:


# Logistic Regression


# In[289]:


feature_cols = ['age', 'sibsp','fare','sex_female', 'sex_male', 'who_child', 'who_man', 'who_woman', 'alone_False', 'alone_True','pclass_1.0','pclass_2.0','pclass_3.0']
Y = df_1['survived']


# In[290]:


from sklearn.linear_model import LogisticRegression


# In[291]:


df_1.iloc[:,0]


# In[275]:


model_logistic = LogisticRegression()


# In[292]:


x_train, x_test, y_train, y_test = train_test_split(df_1.iloc[:,1:], df_1.iloc[:,0], test_size = 0.2, random_state = 42) 


# In[293]:


df_1.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[294]:


model_logistic.fit(x_train, y_train)


# In[295]:


predictions = model_logistic.predict(x_test)


# In[296]:


score = model_logistic.score(x_test, y_test)
print(score)


# In[297]:


# Calculating Confusion matrix for the model
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, predictions)
print(matrix)


# In[212]:


# Decision Tree 


# In[322]:


from sklearn import tree


# In[323]:


model = tree.DecisionTreeClassifier(criterion='entropy')


# In[324]:


model.fit(x_train, y_train)


# In[325]:


prediction = model.predict(x_test)


# In[326]:


score = model.score(x_test, y_test)
print(score)


# The Decision Tree model appears to be more accurate at predicting survival compared to the Logistic Regression model, with an accuracy of 85% compared to 77.5 for the Logistic model.

# In[327]:


# Random Forest


# In[328]:


from sklearn.ensemble import RandomForestClassifier


# In[329]:


model= RandomForestClassifier(n_estimators=1000)


# In[330]:


model.fit(x_train, y_train)


# In[331]:


prediction = model.predict(x_test)


# In[332]:


score = model.score(x_test, y_test)
print(score)


# In[225]:


# GBM


# In[335]:


from sklearn.ensemble import GradientBoostingClassifier


# In[336]:


gb_clf = GradientBoostingClassifier(n_estimators=5, random_state=0)


# In[337]:


gb_clf.fit(x_train, y_train)


# In[338]:


prediction = gb_clf.predict(x_test)


# In[339]:


score = gb_clf.score(x_test, y_test)
print(score)


# Gradient Boosting classifier and Random Forest yield an accuracy score of 75%

# In[ ]:




