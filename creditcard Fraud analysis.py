#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




# In[2]:


pip install -U imbalanced-learn


# In[3]:


pip install -U ydata-profiling


# In[4]:


from ydata_profiling import ProfileReport


# In[ ]:





# In[ ]:





# In[5]:


#Loading the data from a csv file using pandas
#providing link of source file location
data=pd.read_csv("D:\panda\creditcard fraud analysis\creditcard.csv")


# # display top 5 rows of dataset

# In[6]:


data.head()


# In[7]:


pd.options.display.max_columns= None


# In[8]:


creditcard_profile=ProfileReport(data,title="credit card profile report")
creditcard_profile


# In[9]:


creditcard_profile.to_file("your_report.html")


# In[ ]:





# In[10]:


data.head()


# # display last 5 rows of dataset

# In[11]:


data.tail()


# # Find the shape of dataset
# 

# In[12]:


data.shape


# In[13]:


print("Number of rows:",data.shape[0])
print("Number of columns:",data.shape[1])


# In[14]:


data.describe()


# # get  information about dataset like number of rows. number of columns,datatypes of each column and memory requirement

# In[15]:


data.info()


# # check the null values in dataset

# In[16]:


data.isnull().sum()


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


sc=StandardScaler()
data["Amount"]=sc.fit_transform(pd.DataFrame(data["Amount"]))


# In[19]:


data.head()


# In[20]:


data=data.drop(["Time"],axis=1)


# In[21]:


data


# In[22]:


data.shape


# In[23]:


#to check if any duplicates
data.duplicated().any()


# In[24]:


data=data.drop_duplicates()


# In[25]:


data.shape


# In[26]:


#below number of duplicates are dropprd
284807-275663


# # not handling imbalanced

# In[27]:


data["Class"].value_counts()


# In[28]:


sns.countplot(data["Class"])


# In[29]:


corrmat=data.corr()
corrmat


# In[30]:


import matplotlib.pyplot as plt


# In[31]:


corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# # storing feature matrix in X and terget variable in Y

# In[ ]:





# In[32]:


#X=data.drop("Class",axis=1)
#y=data["Class"]


# # splitting data into train and test dataset

# In[33]:


#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# # Handling imbalanced dataset

# In[34]:


#undersampling
#oversampling


# #undersampling

# In[35]:


normal=data[data["Class"]==0]
fraud=data[data["Class"]==1]


# In[36]:


normal.shape


# In[37]:


fraud.shape


# In[38]:


normal_sample=normal.sample(n=473)


# In[39]:


normal_sample.shape


# In[40]:


new_data=pd.concat([normal_sample,fraud],ignore_index=True)


# In[41]:


new_data["Class"].value_counts()


# In[42]:


new_data.head()


# In[43]:


X=new_data.drop("Class",axis=1)
y=new_data["Class"]


# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# # Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)


# In[46]:


y_pred1=log.predict(X_test)


# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


accuracy_score(y_test,y_pred1)


# In[49]:


accuracy_score(y_test,y_pred1)


# In[50]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[51]:


precision_score(y_test,y_pred1)


# In[52]:


precision_score(y_test,y_pred1)


# In[53]:


recall_score(y_test,y_pred1)


# In[54]:


recall_score(y_test,y_pred1)


# In[55]:


f1_score(y_test,y_pred1)


# In[56]:


f1_score(y_test,y_pred1)


# # Decision tree classifier

# In[57]:


from sklearn.tree import DecisionTreeClassifier


# In[58]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[59]:


y_pred2=dt.predict(X_test)


# In[60]:


accuracy_score(y_test,y_pred2)


# In[61]:


precision_score(y_test,y_pred2)


# In[62]:


recall_score(y_test,y_pred2)


# In[63]:


f1_score(y_test,y_pred2)


# # Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[65]:


y_pred3=rf.predict(X_test)


# In[66]:


accuracy_score(y_test,y_pred3)


# In[67]:


precision_score(y_test,y_pred3)


# In[68]:


recall_score(y_test,y_pred3)


# In[69]:


f1_score(y_test,y_pred3)


# In[70]:


final_data=pd.DataFrame({"Models":["LR","DT","RF"],"ACC":[accuracy_score(y_test,y_pred1)*100,
                                              accuracy_score(y_test,y_pred2)*100,
                                              accuracy_score(y_test,y_pred3)*100]})


# In[71]:


final_data


# In[72]:


sns.barplot(final_data["Models"],final_data["ACC"])


# #Oversampling

# In[73]:


X=data.drop("Class",axis=1)
y=data["Class"]


# In[74]:


X.shape


# In[75]:


y.shape


# In[76]:


from imblearn.over_sampling import SMOTE


# In[77]:


X_res,y_res=SMOTE().fit_resample(X,y)


# In[78]:


y_res.value_counts()


# In[79]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)


# # Logistic REgression

# In[80]:


log=LogisticRegression()
log.fit(X_train,y_train)


# In[81]:


y_pred1=log.predict(X_test)


# In[82]:


accuracy_score(y_test,y_pred1)


# In[83]:


precision_score(y_test,y_pred1)


# In[84]:


recall_score(y_test,y_pred1)


# In[85]:


f1_score(y_test,y_pred1)


# # DecisionTreeClassifier

# In[86]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[87]:


y_pred2=dt.predict(X_test)


# In[88]:


accuracy_score(y_test,y_pred2)


# In[89]:


precision_score(y_test,y_pred2)


# In[90]:


recall_score(y_test,y_pred2)


# In[91]:


f1_score(y_test,y_pred2)


# # RandomForestClassifer

# In[92]:


rf=RandomForestClassifier()


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_pred3=rf.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred2)


# In[ ]:


precision_score(y_test,y_pred2)


# In[ ]:


recall_score(y_test,y_pred2)


# In[ ]:


f1_score(y_test,y_pred2)


# In[ ]:


final_data=pd.DataFrame({"Models":["LR","DT","RF"],"ACC":[accuracy_score(y_test,y_pred1)*100,
                                              accuracy_score(y_test,y_pred2)*100,
                                              accuracy_score(y_test,y_pred3)*100]})


# In[ ]:


final_data


# In[ ]:


sns.barplot(final_data["Models"],final_data["ACC"])


# # Save the model

# In[ ]:


rf1=RandomForestClassifier()


# In[ ]:


rf1.fit(X_res,y_res)


# In[ ]:


import joblib


# In[ ]:


joblib.dump(rf1,"credit_card_model")


# In[ ]:


model=joblib.load("credit_card_model")


# In[ ]:


pred=model.predict([[1,1,1,1,1,1,1,1,90890,1,1,1,1,1,1,1,1,1,89,1,1,1,1,1,1,1,1,9098,1]])


# In[ ]:


if pred==0:
    print("Normal Transaction")
else:
    print("Fraudulent Transaction")


# In[ ]:




