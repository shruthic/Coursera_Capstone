#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection
# 
# In this project you will predict fraudulent credit card transactions with the help of Machine learning models. Please import the following libraries to get started.

# In[6]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from scipy import interp
import itertools


# ## Exploratory data analysis

# In[8]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[9]:


#Missing Value Analysis
df.isnull().values.any()


# In[10]:


#observe the different feature type present in the data
df.info()


# In[11]:


df.describe() #Describing the distribution of the each column


# Here we will observe the distribution of our classes

# In[12]:


classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100


# # EXPLORATORY DATA ANALYSIS

# ###### Percentage of regular and fraud Transactions

# In[13]:


print('Percentage of Normal Transactions: {}%'.format(round(df.Class.value_counts()[0]/len(df) * 100.0,2)))
print('Percentage of Frauds: {}%'.format(round(df.Class.value_counts()[1]/len(df) * 100.0,2)))


# ###### Normal and fraud Dataset Creation

# In[14]:


fraud = df[df['Class']==1]
normal = df[df['Class']==0]
fraud.head()


# In[15]:


# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
#sns.barplot(x= 'Normal' y= 'Fraud', data = df)
LABELS = ["Normal", "Fraud"]
count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[16]:


# Create a scatter plot to observe the distribution of classes with time
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time vs Amount divided by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time given in Seconds')
plt.ylabel('Amount')
plt.show()


# In[17]:


# Create a scatter plot to observe the distribution of classes with Amount
plt.scatter(df.Class, df.Amount)
plt.xlabel('Class')
plt.ylabel('Amount')
plt.show()


# In[18]:


# Let us further analyze the variable "Amount"
plt.boxplot(df['Amount'], labels = ['Boxplot'])
plt.ylabel('Transaction amount')
plt.plot()

amount = df[['Amount']].sort_values(by='Amount')


# In[19]:


q1, q3 = np.percentile(amount,[25,75])
q1,q3


# In[20]:


iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)
print('# outliers below the lower bound: ', amount[amount['Amount'] < lower_bound].count()[0],
     ' ({:.4}%)'.format(amount[amount['Amount'] < lower_bound].count()[0] / amount['Amount'].count() * 100))
print('# outliers above the upper bound: ', amount[amount['Amount'] > upper_bound].count()[0],
      ' ({:.4}%)'.format(amount[amount['Amount'] > upper_bound].count()[0] / amount['Amount'].count() * 100))


# If we delete this outliers we will be losing about 11.2% of Data

# In[21]:


#Checking Corelation with the Heatmap
heatmap = sns.heatmap(df.corr(method='spearman'),cmap='coolwarm',robust = True)


# In[22]:


heatmap = sns.heatmap(df.corr(method='pearson'),cmap='coolwarm',robust = True)


# Because of PCA Transformation we have not found any Corelation Issues

# In[23]:


# Drop unnecessary columns
df = df.drop(['Time'], axis=1)


# Since Time Variable will not contribute much to our model we are dropping it for better efficiency

# In[24]:


df.shape


# ##### DATA SCALING

# In[25]:


from sklearn.preprocessing import StandardScaler
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.head()


# We can see the situation of class imbalance clearly

# ### Splitting the data into train & test data

# In[26]:


y= df['Class']
X = df.drop(columns=['Class'])


# In[27]:


X.head()


# In[28]:


y.head()


# In[29]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[30]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state=42)


# ##### Preserve X_test & y_test to evaluate on the test data once you build the model

# ### MODEL BUILDING

# In[31]:


import statsmodels.api as sm


# In[32]:


# Building simple Logistic regression model without balancung the result
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[33]:


#Feature selection using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[34]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # we are running the RFE as 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[35]:


rfe.support_


# In[36]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[37]:


col = X_train.columns[rfe.support_]


# In[38]:


X_train.columns[~rfe.support_]


# In[39]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[40]:


#Acess the model with the stats madel
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]


# In[41]:


y_train_pred_final = pd.DataFrame({'Class':y_train.values, 'Class_Prob':y_train_pred})
y_train_pred_final.head()


# In[42]:


y_train_pred_final['predicted'] = y_train_pred_final.Class_Prob.map(lambda x: 0 if x < 0.01 else 1)
y_train_pred_final.tail()


# In[43]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.predicted )
print(confusion)


# In[44]:


#Checking the Accuracy
print(metrics.accuracy_score(y_train_pred_final.Class, y_train_pred_final.predicted))


# ## Model building with balancing Classes
# 
# ##### Perform class balancing with :
# - Random Oversampling
# - SMOTE
# - ADASYN
# - Random Undersampling

# In[45]:


X_train.head()


# ## Random Undersampling

# In this method, you have the choice of selecting fewer data points from the majority class for your model-building process. In case you have only 500 data points in the minority class, you will also have to take 500 data points from the majority class; this will make the classes somewhat balanced. However, in practice, this method is not effective because you will lose over 99% of the original data.

# In[46]:


rus = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)
X_rus, y_rus = rus.fit_resample(X_train, y_train)


# In[ ]:


X_rus.shape,y_rus.shape


# In[ ]:


y_rus.head()


# In[47]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_rus)[0], Counter(y_rus)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[ ]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_rus)))


# ## Random Oversampling

# Random oversampling involves randomly minority points from the minority to group to match the length of the majority class. The process is entirely randowm it takes few rows from the minority class and adds up

# In[49]:


ros = RandomOverSampler(sampling_strategy='auto', random_state=48)
X_ros, y_ros = ros.fit_resample(X_train, y_train)


# In[50]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_ros)[0], Counter(y_ros)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[51]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_ros)))


# ## Synthetic Minority Over-Sampling Technique (SMOTE)

# In this process, you can generate new data points, which lie vectorially between two data points that belong to the minority class. These data points are randomly chosen and then assigned to the minority class. This method uses K-nearest neighbours to create random synthetic samples

# In[52]:


smote = SMOTE(sampling_strategy='auto', random_state=48)
X_smote, y_smote = smote.fit_resample(X_train, y_train)


# In[53]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_smote)[0], Counter(y_smote)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[54]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_smote)))


# ## ADASyn(Adaptive synthesis)

# This is similar to SMOTE, with a minor change in the generation of synthetic sample points for minority data points. For a particular data point, the number of synthetic samples that it will add will have a density distribution, whereas, for SMOTE, the distribution will be uniform. The aim here is to create synthetic data for minority examples that are harder to learn, rather than the easier ones

# In[55]:


from imblearn.over_sampling import ADASYN


# In[57]:


ads = ADASYN(sampling_strategy='auto', random_state=48)
X_ads, y_ads = ads.fit_resample(X_train, y_train)


# In[58]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_ads)[0], Counter(y_ads)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[59]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_ads)))


# We will create 2d plot to visualize the transformed data

# In[61]:


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#8c564b', '#FF7F0E']
    markers = ['v', '^']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# In[62]:


pca = PCA(n_components=2)
X_ros_pca = pca.fit_transform(X_ros)
X_smote_pca = pca.fit_transform(X_smote)
X_ads_pca = pca.fit_transform(X_ads)
X_rus_pca = pca.fit_transform(X_rus)


# In[63]:


plot_2d_space(X_ros_pca, y_ros, 'Balanced dataset PCA_transformed using random oversampling')
plot_2d_space(X_smote_pca, y_smote, 'Balanced dataset PCA_transformed using SMOTE')


# In[64]:


plot_2d_space(X_ads_pca, y_ads, 'Balanced dataset PCA_transformed using adaptive synthesis')
plot_2d_space(X_rus_pca, y_rus, 'Balanced dataset PCA_transformed using random undersampling')


# ## Logistic regression on random undersampling data

# In[65]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_rus, y_rus)


# In[66]:


lr_predict = lr.predict(X_test)


# In[67]:


lr_predict


# ## Prediction scores

# In[68]:


from sklearn.metrics import accuracy_score, recall_score, confusion_matrix,roc_auc_score
from matplotlib import pyplot

lr_accuracy = accuracy_score(y_test, lr_predict)
lr_recall = recall_score(y_test, lr_predict)
lr_cm = confusion_matrix(y_test, lr_predict)
lr_auc = roc_auc_score(y_test, lr_predict)

print("Accuracy: {:.4%}".format(lr_accuracy))
print("Recall: {:.4%}".format(lr_recall))
print("ROC AUC: {:.4%}".format(lr_auc))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on random oversampling data¶

# In[69]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_ros, y_ros)


# In[70]:


lr_predict_ros = lr.predict(X_test)


# In[71]:


lr_accuracy_ros = accuracy_score(y_test, lr_predict_ros)
lr_recall_ros = recall_score(y_test, lr_predict_ros)
lr_cm_ros = confusion_matrix(y_test, lr_predict_ros)
lr_auc_ros = roc_auc_score(y_test, lr_predict_ros)

print("Accuracy: {:.4%}".format(lr_accuracy_ros))
print("Recall: {:.4%}".format(lr_recall_ros))
print("ROC AUC: {:.4%}".format(lr_auc_ros))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on SMOTE oversampling data

# In[72]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_smote, y_smote)


# In[73]:


lr_predict_smote = lr.predict(X_test)


# In[74]:


lr_accuracy_smote = accuracy_score(y_test, lr_predict_smote)
lr_recall_smote = recall_score(y_test, lr_predict_smote)
lr_cm_smote = confusion_matrix(y_test, lr_predict_smote)
lr_auc_smote = roc_auc_score(y_test, lr_predict_smote)

print("Accuracy: {:.4%}".format(lr_accuracy_smote))
print("Recall: {:.4%}".format(lr_recall_smote))
print("ROC AUC: {:.4%}".format(lr_auc_smote))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on Adasyn oversampling data

# In[75]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_ads, y_ads)


# In[76]:


lr_predict_ads = lr.predict(X_test)


# In[77]:


lr_accuracy_ads = accuracy_score(y_test, lr_predict_ads)
lr_recall_ads = recall_score(y_test, lr_predict_ads)
lr_cm_ads = confusion_matrix(y_test, lr_predict_ads)
lr_auc_ads = roc_auc_score(y_test, lr_predict_ads)

print("Accuracy: {:.4%}".format(lr_accuracy_ads))
print("Recall: {:.4%}".format(lr_recall_ads))
print("ROC AUC: {:.4%}".format(lr_auc_ads))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# perfom cross validation on the X_train & y_train to create:
# X_train_cv
# X_test_cv
# y_train_cv
# y_test_cv

# ###### Similarly explore other algorithms on balanced dataset by building models like:

# ###### KNN Random Forest XGBoost

# Apart from logistic regression let us explore other option since it a classification problem logistic regression is prefferred over all other

# In[78]:


from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp


# Using SMOTE we got: Accuracy: 97.6780% Recall: 87.8378% ROC AUC: 92.7664% Also we are not loosing any information hence we will use this technique further

# # Random Forest

# Let's first fit a random forest model with default hyperparameters.

# one of the most popular algorithms in machine learning. Random forests use a technique known as bagging, which is an ensemble method. So before diving into random forests, let's first understand ensembles.

# In[79]:



# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[80]:


# fit
rfc.fit(X_smote,y_smote)


# In[81]:


# Making predictions
predictions = rfc.predict(X_test)


# In[82]:


# Making predictions
predictions = rfc.predict(X_test)


# In[83]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[84]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# In[85]:


print(accuracy_score(y_test,predictions))


# let's now look at the list of hyperparameters which we can tune to improve model performance.

# In[86]:


model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[87]:


# Fit on training data
model.fit(X_smote, y_smote)


# In[88]:


# Making predictions
predictions = model.predict(X_test)


# In[89]:


print(classification_report(y_test,predictions))


# In[90]:


# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[91]:


from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)


# In[92]:


roc_value


# In[93]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[94]:


fpr, tpr, thresholds = metrics.roc_curve( y_test, rf_probs, drop_intermediate = False )

draw_roc(y_test, rf_probs)


# ### XG BOOST

# In[96]:


import xgboost as xgb


# In[97]:


from xgboost import XGBClassifier
tree_range = range(2, 30, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_smote,y_smote)
    score1.append(xgb.score(X_smote,y_smote))
    score2.append(xgb.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# As we can see accuracy is increasing for the test and stabilizes at one point

# In[98]:


xgb=XGBClassifier(n_estimators=18)
xgb.fit(X_smote,y_smote)
print('Accuracy of XGB on the testing dataset is :{:.3f}'.format(xgb.score(X_test,y_test)))


# In[99]:


# we got a 98% score using xgboost


# In[100]:


print(xgb.feature_importances_)


# In[101]:


pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()


# In[102]:


from xgboost import plot_importance
plot_importance(xgb)
pyplot.show()


# #### 3. Cross-Validation:

# The following figure illustrates k-fold cross-validation with k=4. There are some other schemes to divide the training set, we'll look at them briefly later.

# ### K-Fold Cross Validation

# It is a statistical technique which enables us to make extremely efficient use of available data It divides the data into several pieces, or 'folds', and uses each piece as test data one at a time

# In[103]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[104]:


xgb=XGBClassifier(n_estimators=18)
scores = cross_val_score(xgb, X_smote, y_smote, scoring='r2', cv=5)
scores


# In[105]:


# the other way of doing the same thing (more explicit)

# create a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores_1 = cross_val_score(xgb, X_smote, y_smote, scoring='r2', cv=folds)
scores_1


# We used several methods to predict the default the best result we got by using XGboost on data which was sampled using SMOTE the Accuracy of XGB on the testing dataset is :0.981. Also the important features are:V4,V14,V12,V16,V11. Also by performing logistic regression we got a good score of Accuracy: 97.6780% Recall: 87.8378% ROC AUC: 92.7664% For classification model.

# In[ ]:




