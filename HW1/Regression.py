#import modules and load file

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import xticks
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.linear_model import ElasticNet, ElasticNetCV

df = pd.DataFrame(pd.read_csv("/Users/ananyasharma/Downloads/usc-dsci552-32415d-spring2021/used_car_dataset.csv"))

#clean data
df = df.dropna(how='any',axis=0)

#show cars by manufacturers
Ford = df.loc[df['manufacturer'] == 'ford', 'price']
Subaru = df.loc[df['manufacturer'] == 'subaru', 'price']
fig, ax = plt.subplots(figsize=(20,10))
ax.hist(Subaru, color='#00ff00', alpha=1.0, bins=100, range = [0, 498420],
       label='Manufacturer is Subaru')
ax.hist(Ford, color='#0000ff', alpha=0.5, bins=100, range = [0, 498420],
       label='Manufacturer is Ford')
plt.xlabel('price', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Price Distribution by Manufacturer Type', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend()
plt.show()

print('The average price is {}'.format(round(Subaru.mean(), 2)), 'if manufacturer is Subaru');
print('The average price is {}'.format(round(Ford.mean(), 2)), 'if manufacturer is Ford')


#frequent cost of a car
plt.subplot(1, 2, 1)
(df['price']).plot.hist(bins=50, figsize=(20, 10), edgecolor = 'white', range = [0, 402498])
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)


#hot encode categorical variables
df_onehot = df.copy()
df_onehot = pd.get_dummies(df_onehot, columns=['manufacturer','condition','cylinders','fuel','transmission','type','paint_color','F4'])
(df_onehot)


#correlation matrix
corrMatrix1 = df_onehot.corr()
sns.heatmap(corrMatrix1, annot=True)
plt.show()

#show correlation for price 
corrMatrix1["price"].sort_values(ascending=False)

#scale the data
df_onehot1 = df_onehot.copy()
scaler = preprocessing.MinMaxScaler()
names = df_onehot1.columns
d2 = scaler.fit_transform(df_onehot1)
scaled_df = pd.DataFrame(d2, columns=names)
scaled_df

#histograms for each feature
scaled_df.hist(bins=70, figsize=(20,15))
plt.show()


#split data into training and testing
Y = scaled_df['price']
X = scaled_df.drop(['price'], axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)
# print the data
x_train

#model 1, simple linear regression with only feature year

x_train_1 = x_train['year']
x_train_1 = x_train_1.to_numpy()
x_train_1 = x_train_1.reshape((-1,1))
print(x_train_1.shape)
y_train_1 = y_train.to_numpy()
y_train_1 = y_train_1.reshape((-1,1))
print(y_train_1.shape)

x_test_1 = x_test['year']
x_test_1 = x_test_1.to_numpy()
x_test_1 = x_test_1.reshape((-1,1))
x_test_1.shape


#fitting and training model
model = LinearRegression().fit(x_train_1, y_train)
r_sq = model.score(x_train_1, y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred_1 = model.predict(x_test_1)
print('predicted response:', y_pred_1, sep='\n')
y_pred_1 = pd.Series(y_pred_1)
# model evaluation
rmse = mean_squared_error(y_test, y_pred_1)
r2 = r2_score(y_test, y_pred_1)

# printing values
print('Slope:' ,model.coef_)
print('Intercept:', model.intercept_)
print('Root mean squared error: ', np.sqrt(rmse))
print('mean squared error: ', rmse)
print('R2 score: ', r2)

# Plot outputs

plt.figure(figsize=(12, 6))
plt.plot(scaled_df['year'], scaled_df['price'], 'o')   
plt.plot(x_test_1, y_pred_1, color='blue', linewidth=3)
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()


#model 2 with feature F2

x_train_2 = x_train['F2']
x_train_2 = x_train_2.to_numpy()
x_train_2 = x_train_2.reshape((-1,1))
print(x_train_2.shape)

y_train_2 = y_train.to_numpy()
y_train_2 = y_train_2.reshape((-1,1))
print(y_train_2.shape)

x_test_2 = x_test['F2']
x_test_2 = x_test_2.to_numpy()
x_test_2 = x_test_2.reshape((-1,1))
x_test_2.shape


#fitting and training model
model_2 = LinearRegression().fit(x_train_2, y_train)
r_sq_2 = model_2.score(x_train_2, y_train)
print('coefficient of determination:', r_sq_2)
print('intercept:', model_2.intercept_)
print('slope:', model_2.coef_)
y_pred_2 = model_2.predict(x_test_2)
print('predicted response:', y_pred_2, sep='\n')
y_pred_2 = pd.Series(y_pred_2)

# model evaluation
rmse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

# printing values
print('Slope:' ,model_2.coef_)
print('Intercept:', model_2.intercept_)
print('Root mean squared error: ', np.sqrt(rmse_2))
print(' mean squared error: ', rmse_2)
print('R2 score: ', r2_2)

# Plot outputs
plt.figure(figsize=(12, 6))
plt.plot(scaled_df['F2'], scaled_df['price'], 'o')   
plt.plot(x_test_2, y_pred_2, color='blue', linewidth=3)
plt.xlabel('F2')
plt.ylabel('Price')
plt.show()


#model 3 with feature type_pickup
x_train_3 = x_train['type_pickup']
x_train_3 = x_train_3.to_numpy()
x_train_3 = x_train_3.reshape((-1,1))
print(x_train_3.shape)

y_train_3 = y_train.to_numpy()
y_train_3 = y_train_3.reshape((-1,1))
print(y_train_3.shape)

x_test_3 = x_test['type_pickup']
x_test_3 = x_test_3.to_numpy()
x_test_3 = x_test_3.reshape((-1,1))
x_test_3.shape

#fitting and training model
model_3 = LinearRegression().fit(x_train_3, y_train)
r_sq_3 = model_3.score(x_train_3, y_train)
print('coefficient of determination:', r_sq_3)
print('intercept:', model_3.intercept_)
print('slope:', model_3.coef_)
y_pred_3 = model_3.predict(x_test_3)
print('predicted response:', y_pred_3, sep='\n')
y_pred_3 = pd.Series(y_pred_3)

# model evaluation
rmse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)

# printing values
print('Slope:' ,model_3.coef_)
print('Intercept:', model_3.intercept_)
print('Root mean squared error: ', np.sqrt(rmse_3))
print('mean squared error: ', rmse_3)
print('R2 score: ', r2_3)


# Plot outputs
plt.figure(figsize=(12, 6))
plt.plot(scaled_df['type_pickup'], scaled_df['price'], 'o')   
plt.plot(x_test_3, y_pred_3, color='blue', linewidth=3)
plt.xlabel('Type_Pickup')
plt.ylabel('Price')
plt.show()


#model 4, multiple linear regression with features year and F2

x_train_4 = x_train[['year','F2']]
x_test_4 = x_test[['year','F2']]

#fitting and training
model_4 = LinearRegression().fit(x_train_4, y_train)
r_sq_4 = model_4.score(x_train_4, y_train)
print('coefficient of determination:', r_sq_4)
print('intercept:', model_4.intercept_)
print('slope:', model_4.coef_)
y_pred_4 = model_4.predict(x_test_4)
print('predicted response:', y_pred_4, sep='\n')

# model evaluation
rmse_4 = mean_squared_error(y_test, y_pred_4)
r2_4 = r2_score(y_test, y_pred_4)

# printing values
print('Slope:' ,model_4.coef_)
print('Intercept:', model_4.intercept_)
print('Root mean squared error: ', np.sqrt(rmse_4))
print('mean squared error: ', rmse_4)
print('R2 score: ', r2_4)

#Plot graph
sns.regplot(x=y_pred_4, y=y_test,ci=None,scatter_kws={"color": "teal"}, line_kws={"color": "orange"})

#model 5 with features year, F2 and type_pickup
x_train_5 = x_train[['year','F2','type_pickup']]
x_test_5 = x_test[['year','F2','type_pickup']]

model_5 = LinearRegression().fit(x_train_5, y_train)
r_sq_5 = model_5.score(x_train_5, y_train)
print('coefficient of determination:', r_sq_5)
print('intercept:', model_5.intercept_)
print('slope:', model_5.coef_)
y_pred_5 = model_5.predict(x_test_5)
print('predicted response:', y_pred_5, sep='\n')


# model evaluation
rmse_5 = mean_squared_error(y_test, y_pred_5)
r2_5 = r2_score(y_test, y_pred_5)

# printing values
print('Slope:' ,model_5.coef_)
print('Intercept:', model_5.intercept_)
print('mean squared error: ', rmse_5)
print('Root mean squared error: ', np.sqrt(rmse_5))
print('R2 score: ', r2_5)

#Plot graph
sns.regplot(x=y_pred_5, y=y_test,ci=None,scatter_kws={"color": "teal"}, line_kws={"color": "orange"})


#model 6 with all features, fitting and training
model_6 = LinearRegression().fit(x_train, y_train)
r_sq_6 = model_6.score(x_train, y_train)
print('coefficient of determination:', r_sq_6)
print('intercept:', model_6.intercept_)
print('slope:', model_6.coef_)
y_pred_6 = model_6.predict(x_test)
print('predicted response:', y_pred_6, sep='\n')


# model evaluation
rmse_6 = mean_squared_error(y_test, y_pred_6)
r2_6 = r2_score(y_test, y_pred_6)

# printing values
print('Slope:' ,model_6.coef_)
print('Intercept:', model_6.intercept_)
print('Root mean squared error: ', np.sqrt(rmse_6))
print('mean squared error: ', rmse_6)
print('R2 score: ', r2_6)

#plotting graph
sns.regplot(x=y_pred_6, y=y_test,ci=None,scatter_kws={"color": "teal"}, line_kws={"color": "orange"})


#model 7 ridge regression
#Ridge Regression
#finding optimum alpha value

alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1,2]
for a in alphas:
 find_model = Ridge(alpha=a).fit(X,Y) 
 find_score = find_model.score(X, Y)
 find_pred_y = find_model.predict(X)
 find_mse = mean_squared_error(Y, find_pred_y) 
 print("Alpha:{0:.6f}, R2:{1:.15f}, MSE:{2:.15f}, RMSE:{3:.15f}".format(a, find_score, find_mse, np.sqrt(find_mse)))

#fitting and training
model_ridge = Ridge(alpha=0.000001).fit(x_train,y_train)
ypred_ridge = model_ridge.predict(x_test)
score_ridge = model_ridge.score(x_test,y_test)
mse_ridge = mean_squared_error(y_test,ypred_ridge)
print("R2:{0:.10f}, MSE:{1:.15f}, RMSE:{2:.10f}"
   .format(score_ridge, mse_ridge,np.sqrt(mse_ridge)))
   
 
 
#Plot graph
x_ax = range(len(x_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred_ridge, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()



#model 8 LASSO regression

#find optimum alpha value
lassomodel1 = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
lassomodel1.fit(x_train, y_train)
# summarize chosen configuration
print('alpha: %f' % lassomodel1.alpha_)

#fit and train
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(lassomodel, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

lassomodel.fit(x_train,y_train)
ypredlasso = lassomodel.predict(x_test)
print('MSE: %.15f,R2 : %.7f, RMSE : %.15f' %(mean_squared_error(y_test,ypredlasso),lassomodel.score(x_train,y_train),np.sqrt(mean_squared_error(y_test,ypredlasso))))



#model 9 , ElasticNet

#find optimum hyperparameter values
ratios = arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
modelen1 = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
modelen1.fit(x_train, y_train)
# summarize chosen configuration
print('alpha: %f' % modelen1.alpha_)
print('l1_ratio_: %f' % modelen1.l1_ratio_)

 define model
modelen2 = ElasticNet(alpha=0.000100, l1_ratio=0)
# define model evaluation method
cv1 = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores2 = cross_val_score(modelen2, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv1, n_jobs=-1)
# force scores to be positive
scores22 = absolute(scores2)
print('Mean MAE: %.3f (%.3f)' % (mean(scores22), std(scores22)))

modelen2.fit(x_train,y_train)
ypreden2 = modelen2.predict(x_test)
print('MSE: %.15f,R2 : %.7f, RMSE: %.7f' %(mean_squared_error(y_test,ypreden2),modelen2.score(x_train,y_train),np.sqrt(mean_squared_error(y_test,ypreden2))))

#plot
x_ax1 = range(len(x_test))
plt.scatter(x_ax1, y_test, s=5, color="blue", label="original")
plt.plot(x_ax1, ypreden2,lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

