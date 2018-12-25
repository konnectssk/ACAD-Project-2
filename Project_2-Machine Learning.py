
# coding: utf-8

# # Project 2

# #### In this project you are going to predict the overall rating of soccer player based on their attributessuch as 'crossing','finishing etc.
# The dataset you are going to use is from European Soccer Database
# (https://www.kaggle.com/hugomathien/soccer) has more than 25,000 matches and more than
# 10,000 players for European professional soccer seasons from 2008 to 2016.
# Download the data in the same folder and run the following commmand to get it in the environment

# ## Import data

# In[2]:


import sqlite3 #for using sql query 


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor # modelling 
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split # validation
from sklearn.metrics import mean_squared_error  # metrics
from math import sqrt # mathamtical calculation


# In[ ]:


cnx = sqlite3.connect('database.sqlite') #create a connection object for sql db
# database.sqlite IS SQLITE file downloaded from KAGGLE.COM used for importing the dataset


# In[ ]:


df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx) #saving data from selelct query in the form of dataframe


# In[ ]:


df.head()


# ## Exploratery data analysis

# In[ ]:


df.dtypes #TO chack datatypes of the all the columns


# In[ ]:


df.describe() #for numeric variable to get stats 


# In[ ]:


df.shape   #nrows and ncols


# In[ ]:


df.groupby('player_fifa_api_id').count()


# In[ ]:


df['player_fifa_api_id'].value_counts() # to check ids with no of time occurance


# ### Missing data calculation

# In[ ]:


missing= df.isnull().sum() #missing values in the dataframe


# In[ ]:


missing


# In[ ]:


percent_of_missing= (df.isnull().sum()/df.isnull().count())*100 


# In[ ]:


percent_of_missing #is showing percent of the nulls present in every columns


# In[ ]:


df['overall_rating'].max() #max rating of the player 


# In[ ]:


df['overall_rating'].min()


# This dataset has NaN but less than 5% which can be ignored. hence dropped the rows having NaN

# In[ ]:


df1=df.dropna() #dropping all the rows having atleast one NaN


# In[ ]:


print("No of rows and columns of the original data", df.shape)
print("No of rows and columns of the data after NA removed",  df1.shape)


# <h4>Since ids have no impact on overall ratings of the player we can remove the columns having player ids</h4>

# In[ ]:


correlations= df.corr()['overall_rating'].sort_values()
correlations


# <h4>Three columns player_api_id, player_fifa_api_id and id are negatively weak correlated and hence removing them from dataset</h4>

# In[ ]:


Target=df['overall_rating']


# In[ ]:


del df1['id']
del df1['player_fifa_api_id']
del df1['player_api_id']


# In[ ]:


df1.shape #after removing three columns (id, player_fifa_api_id, player_api_id) from the dataframe


# In[ ]:


df1.columns   #columns present in the dataframe


# ### To create features using OHE we need to know which are the ones having object as datatype.

# In[ ]:


total_obj=df1.columns[df1.dtypes.values=='object'] #to column having datatype as object 
total_obj


# In[ ]:


#df['positioning'] is numeric
#df['gk_handling']
#df['heading_accuracy']
#df['aggression']
#df['heading_accuracy']
#df['marking']


# In[ ]:


from datetime import datetime
from dateutil import parser
df1.loc[:,'date']= df1['date'].apply(pd.to_datetime) # converting date from object datatype to pandas datetime 


# In[ ]:


df1.loc[:,'month']=df1['date'].apply(lambda x: x.month)
df1.loc[:,'year']=df1['date'].apply(lambda x: x.year)
df1.loc[:,'day']=df1['date'].apply(lambda x: x.day)


# In[ ]:


df1.columns[df1.dtypes.values=='object']  


# In[ ]:


df1.shape


# In[ ]:


df1.columns


# #### features having object datatype will be used to create more features using one hot encoding. 

# In[ ]:


#to check no of categories in the categorical variable

print("categories under preferred foot")
print(df['preferred_foot'].value_counts())
print("-------------------------------------")

print("categories under Attacking work rate")
print(df1['attacking_work_rate'].value_counts())
print("-------------------------------------")

print("categories under defensive work rate")
print(df1['defensive_work_rate'].value_counts())


# In[ ]:


df1['defensive_work_rate'].value_counts().index


# In[ ]:


cols=['o','1','2','3','5','4','6','7','8','9','ormal','0','ean','tocky','es']


# In[ ]:


for ind in cols:
    df1.drop(df1.index[df1['defensive_work_rate']==ind],inplace=True)
        


# In[ ]:


df1['defensive_work_rate'].value_counts()


# In[ ]:


df1['attacking_work_rate'].value_counts().index
col_attack=['norm','stoc','le','y']
for ind in col_attack:
    df1.drop(df1.index[df1['attacking_work_rate']==ind],inplace=True)


# In[ ]:


df1['attacking_work_rate'].value_counts() # After removal of other irrelevant classes


# <h1>Plots and visualizations</h1>

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(df1['overall_rating'])
plt.title("Histogram of overall rating")
plt.show( )           #histogram showing distribution of the dependant variable overall fitting


# In[ ]:


df1.head()


# In[ ]:


plt.figure(figsize=(18,6))
plt.scatter(df['id'],df['overall_rating']) #to check the relationship between id and dependant variable


# In[ ]:


#scatter plot between potential and rating

plt.figure(figsize=(18,10))
plt.subplot(2,2,1)
plt.scatter(df['potential'],df['overall_rating'],color='pink')
plt.xlabel("Potentail of the player")
plt.ylabel("rating of the player")
plt.title("scatter plot between potential and rating")

#dribbling vs overall_ratings
plt.subplot(2,2,2)
plt.scatter(df['dribbling'],df['overall_rating'],color='cyan')
plt.xlabel("dribbling")
plt.ylabel("rating of the player")
plt.title("scatter plot between dribbling and rating")

# Finishing vs overall_ratings
plt.subplot(2,2,3)
plt.scatter(df['finishing'],df['overall_rating'],color='brown')
plt.xlabel("finishing")
plt.ylabel("rating of the player")
plt.title("scatter plot between finishing and rating")

# free kick accuracy vs overall ratings
plt.subplot(2,2,4)
plt.scatter(df['free_kick_accuracy'],df['overall_rating'],color='green')
plt.xlabel("FreeKick accuracy")
plt.ylabel("overall_rating")
plt.title("Free kick accuracu vs overall_ratings")
plt.show()


# In[ ]:


col=list(df1.columns)
cols=col[1:]
fig, axes = plt.subplots(10, 4, figsize=(16, 12))
for i,ax in enumerate(axes.flat):
    if i < 41:
        ax.scatter(df1[cols[i+1]], df1[cols[0]])
        ax.set_title(cols[i+1])
plt.tight_layout()


# In[ ]:


df1.columns


# In[ ]:


df1 =  df1.drop("date", axis = 1) # dropping date columns from the dataframe


# <h1>Feature Selection and One hot encoding</h1>

# In[ ]:


for col in ['preferred_foot', 'attacking_work_rate','defensive_work_rate', 'year', 'month','day']:
    dummies=pd.get_dummies(df1[col],prefix=col)
    df1=df1.join(dummies)
    df1=df1.drop(col,axis=1)


# In[ ]:


df1.columns


# In[ ]:


df1.head()


# In[ ]:


df1.shape


# In[ ]:


df1.dropna(inplace=True)


# <h1>Creating training and test set</h1>

# In[ ]:


y=df1.pop('overall_rating')
x=df1


# In[ ]:


y.head()


# In[ ]:


x.head()


# In[ ]:


#creating training and test set for dependant and independant variable
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)


# In[ ]:


train_x.shape #trainingg set of X


# In[ ]:


test_x.shape  #test set of X


# In[ ]:


train_y.shape    #train set of y


# In[ ]:


test_y.shape    #test set of y


# <h3>Using SKlearn </h3>

# In[ ]:


lm=LinearRegression()  # instantiating Linear regression 
lm.fit(train_x,train_y)  #fitting the training set


# In[ ]:


print(lm.intercept_) # intercept beta0
print(lm.coef_)       #gradient beta1 to beta_n


# In[ ]:


pred_train= lm.predict(test_x) # predicting test set


# In[ ]:


# In order to check the model accuracy we are using metric root mean square error
   
import numpy as np
np.sqrt(mean_squared_error(test_y, pred_train))


# In[ ]:


import seaborn as sns # visualization
sns.set_style("whitegrid") # for background
sns.set_context("poster")
plt.figure(figsize=(16,9)) # size of the figure
plt.scatter(test_y,pred_train) # scatter plot
plt.xlabel("Overall Rating: $Y_i$")
plt.ylabel("Predicted Overall Rating: $\hat{Y}_i$")
plt.title("Overall Rating vs Predicted Overall Rating: $Y_i$ vs $\hat{Y}_i$")
plt.text(40,25, "Comparison between the actual Overall Rating and predicted Overall Rating.", ha='left')
plt.show()


# In[ ]:


sns.regplot(test_y,pred_train, data=df1, fit_reg=True) #Plot test_y and pred_train for Linear Regression Model.


# In[ ]:


# recursive feature elimination
from sklearn.feature_selection import RFE
rfe = RFE(lm,10 )
rfe = rfe.fit(df1,y)
print(rfe.support_)
print(rfe.ranking_)


# 
# <h1>Using Statsmodels OLS package</h1>

# In[ ]:


### stats model 


# In[ ]:


import statsmodels.formula.api as smf 
lm1=smf.ols(formula='train_y ~ train_x',data=train_x).fit()  # fitting stats model
#second method of linear regression modelling(OLS)(ordinary least square method)


# In[ ]:


lm1.summary()


# In[ ]:


lm1.rsquared #to check r squared metric for training set


# In[ ]:


lm1.conf_int()


# In[ ]:


y_pred1=lm1.predict(train_x) #using stat model predict the y


# In[ ]:


np.sqrt(mean_squared_error(train_y, y_pred1))


# <h2>  sklearn K fold cross validation</h2>

# #### We have already seen that using sklearn Lm model we got 2.677 RMSE now we want to validate our model on validation set and using 10 fold cross validation. if our model predicts the rating of soccer player with the same RMSE then our model is not overfitted.

# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score  # cross_validation score

cv_score= cross_val_score(LinearRegression(),x,y,scoring='neg_mean_squared_error', cv=10)


# In[ ]:


cv_score # since we have used k=10 there are 10 cv scores


# <h4>calculating mean and square root to get cv_score</h4>

# In[ ]:


cv_score, cv_score.mean()# MSE is determined by taking mean of cv score 
np.sqrt(cv_score.mean() * -1)# RMSE


# In[ ]:


print("The Root Mean Square Error (RMSE) for the Model is "+ str(np.sqrt(cv_score.mean() * -1)) )

