#!/usr/bin/env python
# coding: utf-8

# # MADUGITAL 

# ## 1. Data Preparation 

# ### Import Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


# ### Data Overview

# In[2]:


data_path = os.path.join("Madugital", "lead_scoring.csv")
print(data_path)


# In[3]:


data = pd.read_csv (data_path)


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.info()


# ### Data Quality Check

# In[7]:


#null data percentage (criteria: > 60% = drop, < 60% = fill)
data.isna().mean() * 100


# In[8]:


#duplicated data checking
names = [
    'Prospect ID',
    'Lead Number',
    'Lead Origin',
    'Lead Source',
    'Do Not Email',
    'Do Not Call',
    'Converted',
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Last Activity',
    'Country',
    'Specialization',
    'How did you hear about Madugital',
    'What is your current occupation',
    'What matters most to you in choosing a product',
    'Search',
    'Magazine',
    'Newspaper Article',
    'Madugital Telegram',
    'Newspaper',
    'Digital Advertisement',
    'Through Recommendations',
    'Receive More Updates About Our Products',
    'Tags',
    'Lead Quality',
    'Update me on Supply Chain Content',
    'Get updates on DM Content',
    'Lead Profile',
    'City',
    'Asymmetrique Activity Index',
    'Asymmetrique Profile Index',
    'Asymmetrique Activity Score',
    'Asymmetrique Profile Score',
    'I agree to pay the amount through cheque',
    'A free copy of Mastering The Interview',
    'Last Notable Activity'
]

data.loc[data[names].duplicated()]


# ##### No data duplicated 

# In[9]:


#filling null data
data['Lead Source'] = data['Lead Source'].fillna('Missing')
data['TotalVisits'] = data['TotalVisits'].fillna(data['TotalVisits'].mean())
data['Page Views Per Visit'] = data['Page Views Per Visit'].fillna(data['Page Views Per Visit'].mean())
data['Last Activity'] = data['Last Activity'].fillna('Missing')
data['Country'] = data['Country'].fillna('Missing')
data['Specialization'] = data['Specialization'].fillna('Missing')
data['How did you hear about Madugital'] = data['How did you hear about Madugital'].fillna('Missing')
data['What is your current occupation'] = data['What is your current occupation'].fillna('Missing')
data['What matters most to you in choosing a product'] = data['What matters most to you in choosing a product'].fillna('Missing')
data['Tags'] = data['Tags'].fillna('Missing')
data['Lead Quality'] = data['Lead Quality'].fillna('Missing')
data['Lead Profile'] = data['Lead Profile'].fillna('Missing')
data['City'] = data['City'].fillna('Missing')
data['Asymmetrique Activity Index'] = data['Asymmetrique Activity Index'].fillna('Missing')
data['Asymmetrique Profile Index'] = data['Asymmetrique Profile Index'].fillna('Missing')
data['Asymmetrique Activity Score'] = data['Asymmetrique Activity Score'].fillna(data['Asymmetrique Activity Score'].mean())
data['Asymmetrique Profile Score'] = data['Asymmetrique Profile Score'].fillna(data['Asymmetrique Profile Score'].mean())

data.isnull().sum()


# In[10]:


#cek outliers dengan Boxplot 
data['Lead Number'].plot(kind='box')
plt.title('Boxplot Lead Number', size=16)
plt.show()

data['Converted'].plot(kind='box')
plt.title('Boxplot Converted', size=16)
plt.show()

data['TotalVisits'].plot(kind='box')
plt.title('Boxplot Total Visits', size=16)
plt.show()

data['Total Time Spent on Website'].plot(kind='box')
plt.title('Boxplot Total Time Spent', size=16)
plt.show()

data['Page Views Per Visit'].plot(kind='box')
plt.title('Boxplot Page Views per Visit', size=16)
plt.show()

data['Asymmetrique Activity Score'].plot(kind='box')
plt.title('Boxplot Activity Score', size=16)
plt.show()

data['Asymmetrique Profile Score'].plot(kind='box')
plt.title('Boxplot Profile Score', size=16)
plt.show()


# In[11]:


#handling outliers 

Q1 = data['TotalVisits'].quantile(0.25)
Q3 = data['TotalVisits'].quantile(0.75)
IQR = Q3 - Q1 
Lwhisker = Q1 - 1.5 * IQR 
Uwhisker = Q3 + 1.5 * IQR 
data['TotalVisits_clipped'] = data['TotalVisits'].clip(Lwhisker, Uwhisker) 

Q1 = data['Page Views Per Visit'].quantile(0.25)
Q3 = data['Page Views Per Visit'].quantile(0.75)
IQR = Q3 - Q1 
Lwhisker = Q1 - 1.5 * IQR 
Uwhisker = Q3 + 1.5 * IQR 
data['Page Views Per Visit_clipped'] = data['Page Views Per Visit'].clip(Lwhisker, Uwhisker) 

Q1 = data['Asymmetrique Activity Score'].quantile(0.25)
Q3 = data['Asymmetrique Activity Score'].quantile(0.75)
IQR = Q3 - Q1 
Lwhisker = Q1 - 1.5 * IQR 
Uwhisker = Q3 + 1.5 * IQR 
data['Asymmetrique Activity Score_clipped'] = data['Asymmetrique Activity Score'].clip(Lwhisker, Uwhisker) 

Q1 = data['Asymmetrique Profile Score'].quantile(0.25)
Q3 = data['Asymmetrique Profile Score'].quantile(0.75)
IQR = Q3 - Q1 
Lwhisker = Q1 - 1.5 * IQR 
Uwhisker = Q3 + 1.5 * IQR 
data['Asymmetrique Profile Score_clipped'] = data['Asymmetrique Profile Score'].clip(Lwhisker, Uwhisker) 


# In[12]:


data['TotalVisits_clipped'].plot(kind='box')
plt.title('Boxplot Total Visits_clipped', size=16)
plt.show()

data['Page Views Per Visit_clipped'].plot(kind='box')
plt.title('Boxplot Page Views per Visit_clipped', size=16)
plt.show()

data['Asymmetrique Activity Score_clipped'].plot(kind='box')
plt.title('Boxplot Asymmetrique Activity Score_clipped', size=16)
plt.show()

data['Asymmetrique Profile Score_clipped'].plot(kind='box')
plt.title('Boxplot Asymmetrique Profile Score_clipped', size=16)
plt.show()


# ## 2. Finding Data Insight 

# In[13]:


#identifying engaged leads 
activity_counts = data.groupby('Last Activity').size().reset_index(name='Count')
activity_counts_sorted = activity_counts.sort_values('Count', ascending=False)
print(activity_counts_sorted)


# #### This insight concludes that: 
# 
# ##### 1. "Email Opened" and "SMS Sent", with total of each 3437 and 2745, are the effective communication channels in capturing lead engagements. 
# ##### 2. "Olark Chat Conversation" and "Page Visited on Website", with total of each 973 and 640, are relatively effective to because leads that engage in live chat or visit specific page have shown active interest 
# ##### 3. "Converted to Lead" which means leads that have been converted to potential customers, with total 428, shows progress that indicate higher level of engagement.

# In[14]:


#website engagement 
page_views_stats = data['Page Views Per Visit'].describe()
print(page_views_stats)

median_value = data['Page Views Per Visit'].median()
print("Median:", median_value)

mode_value = data['Page Views Per Visit'].mode()
print("Mode:", mode_value)


# In[15]:


#lead interest 
total_visits_stats = data['TotalVisits'].describe()
print(total_visits_stats)

median_value = data['TotalVisits'].median()
print("Median:", median_value)

mode_value = data['TotalVisits'].mode()
print("Mode:", mode_value)


# In[16]:


#leads behaviors based on last activities
converted_leads = data[data['Converted'] == 1]
converted_leads_last_activity = converted_leads.groupby('Last Activity')['Converted'].count()
print("Leads Last Activity:")
print(converted_leads_last_activity)


# In[17]:


#leads last activities (1 = buy, 0 = not buy)
pd.crosstab(data['Converted'], data['Last Activity'])


# #### SMS Sent is the most potential last activity of customers who converted. Then, email opened.

# In[18]:


#leads behaviors based on lead
converted_leads = data[data['Converted'] == 1]
converted_leads_lead_source = converted_leads.groupby('Lead Source')['Converted'].count()
print("Leads Lead Source:")
print(converted_leads_lead_source)


# #### Google is the most potential lead source of customers who then converted, then direct traffic

# In[19]:


email_opened = data[data['Last Activity'] == "Email Opened"]
email_opened.groupby(['Last Activity', 'Specialization']).size()


# In[20]:


converted_leads_email_opened = converted_leads[converted_leads['Last Activity'] == "Email Opened"]
converted_leads_email_opened.groupby(['Last Activity','Specialization']).size()


# In[21]:


pd.crosstab(data['Last Activity'] == "Email Opened", data['Specialization'])


# In[22]:


converted_leads.groupby(['Converted', 'What matters most to you in choosing a product']).size()


# #### Most converted to leads think that health matters to them in choosing a product

# In[23]:


converted_leads['City'].value_counts().plot(kind='bar')
plt.title("Behavior based on City")
plt.xlabel("City")
plt.ylabel("Count")
plt.show()


# ### Most converted to leads are in Jakarta

# In[24]:


converted_leads_lead_source = converted_leads.groupby('Specialization')['Converted'].count()
print("Leads Lead Source:")
print(converted_leads_lead_source)


# In[25]:


pd.crosstab(converted_leads['Lead Source'], converted_leads['Specialization'])


# In[26]:


crosstab_leadsource_specialization = pd.crosstab(converted_leads['Lead Source'], converted_leads['Specialization'])
print(crosstab_leadsource_specialization)


# #### Conclusion: 
# #### 1. Direct Traffic and Google are mostly potential for any specialization
# #### 2. "Select" specialization reaches the highest potential. However, it can't be defined. 

# In[27]:


data['What is your current occupation'].unique()


# In[28]:


converted_leads.groupby('What is your current occupation')['Converted'].count()


# In[29]:


data['How did you hear about Madugital'].unique()


# In[30]:


converted_leads.groupby('How did you hear about Madugital')['Converted'].count()


# ## 3. Data Manipulation

# In[31]:


#Scaling
scaler1 = StandardScaler()
std_total_visits = scaler1.fit_transform(data[['TotalVisits_clipped']])
std_page_views = scaler1.fit_transform(data[['Page Views Per Visit_clipped']])
std_asymmetrique_activity_score = scaler1.fit_transform(data[['Asymmetrique Activity Score_clipped']])
std_asymmetrique_profile_score = scaler1.fit_transform(data[['Asymmetrique Profile Score_clipped']])


# In[32]:


data['TotalVisits_clipped_scaled'] = std_total_visits
data['Page Views Per Visit_clipped_scaled'] = std_page_views
data['Asymmetrique Activity Score_clipped_scaled'] = std_asymmetrique_activity_score 
data['Asymmetrique Profile Score_clipped_scaled'] = std_asymmetrique_profile_score


# In[33]:


df = pd.DataFrame(data)
df


# In[34]:


data.info()


# In[35]:


data_for_encode = data.drop(['Prospect ID', 'Lead Number', 'Lead Origin', 'TotalVisits', 'Total Time Spent on Website',
                   'Page Views Per Visit', 'Asymmetrique Activity Score', 'Asymmetrique Profile Score', 'TotalVisits_clipped', 'Page Views Per Visit_clipped',
                   'Asymmetrique Activity Score_clipped', 'Asymmetrique Profile Score_clipped'], 1)
data_for_encode.info()


# In[36]:


#Encoding 

pd.get_dummies(data_for_encode, drop_first=True)


# ## 4. Modelling Preparation 

# In[37]:


final_data = pd.get_dummies(data_for_encode, drop_first=True)
final_data.columns


# In[38]:


train, test = train_test_split(final_data, test_size=0.3, random_state=100)


# In[39]:


train.shape, test.shape


# ## 5. Modelling 

# In[40]:


train.columns


# In[41]:


test.columns


# ### Linear Regression

# In[42]:


X_train = train.drop('Converted', axis=1)
y_train = train['Converted']
X_test = test.drop('Converted', axis=1)
y_test = test['Converted']

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[43]:


X_test[2:3]


# In[44]:


y_test[2:3]


# ### Decision Tree Classifier 

# In[52]:


dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=100)
dt.fit(X_train, y_train)


# ### KMeans

# In[53]:


kmeans = KMeans(n_clusters=5, max_iter=100)
kmeans.fit(X_train)


# ## 6. Modelling Evaluation

# In[54]:


from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, silhouette_score


# In[55]:


#Regression Evaluation 
y_pred = lr.predict(X_test)
mean_squared_error(y_test, y_pred)


# In[56]:


#Classification
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(acc, prec, recall)


# In[57]:


#Clustering
silhouette_score(X_train, kmeans.labels_)

