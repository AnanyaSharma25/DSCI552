import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('/Users/ananyasharma/Downloads/usc-dsci552-section-32415d-spring-2021-ps2/ps2_available_dataset.csv')
#shows top 5 rows 
data.head()
#plot to see how many people need treatment
sns.countplot(x='treatment', data=data)
#plot to see how many people need treatment as per gender
sns.countplot(x='treatment', hue='gender', data=data)
#plot to see how many people need treatment as per blood test
sns.countplot(x='treatment', hue='blood_test', data=data)
#histogram to show age distribution, by dropping all nulls
plt.hist(data['age'].dropna())
#similar histograms for other columns 
plt.hist(data['GeneA'])
plt.hist(data['MeasureA'])
plt.hist(data['TestB'])
plt.hist(data['GeneB'])
plt.hist(data['GeneC'])
#see if data has null value. will return true where it does
data.isnull()
#another visualisation to check null data
sns.heatmap(data.isnull(), cbar=False)



#boxplots are to show two variables wrt to each other
sns.boxplot(data['family_history'], data['age'])


sns.boxplot(data['family_history'], data['treatment'])

sns.boxplot(data['family_history'], data['MeasureA'])


sns.boxplot(data['family_history'], data['TestB'])


sns.boxplot(data['family_history'], data['GeneB'])



sns.boxplot(data['family_history'], data['GeneC'])




#check how many columns have how many null entries
print(data.isnull().sum())
#replace null values with median value
median = data['family_history'].median()
data['family_history'].fillna(median, inplace=True)
#again check
print(data.isnull().sum())
sns.heatmap(data.isnull(), cbar=False)


gender_data = pd.get_dummies(data['gender'], drop_first = True)

bt_data = pd.get_dummies(data['blood_test'], drop_first = True)
data = pd.concat([data, gender_data, bt_data], axis = 1)
data.head()
data["GeneA"] = data["GeneA"].astype('category')
data.dtypes


data["GeneA"] = data["GeneA"].cat.codes
data.head()
data["family_history"] = data["family_history"].astype('category')
data.dtypes
data["family_history"] = data["family_history"].cat.codes

data.drop(['gender', 'blood_test'], axis = 1, inplace = True)
data.head()

#test and train data
y_data = data['treatment']

x_data = data.drop('treatment', axis = 1)
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3)


#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

print(classification_report(y_test_data, predictions))


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test_data, predictions))
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(model.score(x_training_data, y_training_data)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(model.score(x_test_data, y_test_data)))
     
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test_data, model.predict(x_test_data))
fpr, tpr, thresholds = roc_curve(y_test_data, model.predict_proba(x_test_data)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

pAUC = np.trapz(tpr, fpr)

print(pAUC)




