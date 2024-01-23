import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("HR_comma_sep.csv") 
df.head()
df.info()
df.describe()
df.columns
df.shape

count_left=np.count_nonzero(df.left==1)
count_promo=np.count_nonzero(df.promotion_last_5years==1)
count_accident=np.count_nonzero(df.Work_accident==1)
left_percent=float(count_left)/float(df.shape[0])*100
print("we observe number o %s people that left the company"% count_left)
print("we observe percent o %s people that left the company"% left_percent)


#Separating tha target variable and input variables
Y= df['left']
X=df.drop('left',axis=1)

#There are no two variables which are hihgly correlated with each other 

correlation=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True, annot=True,cmap='cubehelix')
plt.title('Correlation between different features')
plt.show()


#How many people are leaving depending on their salary
sns.barplot(x=df.salary,y=df.left,order=['high','low','medium'])
plt.show()

#HR has the highest percentage of people leaving and management has the lowest percentage of people
plt.figure(figsize=(15,8))
sns.barplot(x=df.sales,y=df.left,ci=None)
plt.show()

#HR has the highest percentage of  people leaving and management has the lowest percentage of people
plt.figure(figsize=(5,5))
sns.barplot(x=df.promotion_last_5years,y=df.left,ci=None)
plt.show()

#Dummy variable creation

df_2=pd.get_dummies(df)
encoded=list(df_2.columns)
print(encoded)

#dropping extra variables
df_2=df_2.drop(['salary_medium'],axis=1)
df_2=df_2.drop(['sales_technical'],axis=1)
df_2.head()