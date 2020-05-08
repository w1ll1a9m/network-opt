
#%%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError

data = pd.read_excel('Xtraction Tableau.xlsx')
data.dropna(how = 'all', inplace = True)
data.dropna(subset =['Total Cost (EUR)','Cost/Kg'], inplace = True)
data.drop(['BU','Origin City','Destination Country','Destination City','Total Cost (EUR)'], axis = 1, inplace = True)

# Get the index

indexNames = data[data['Region Cluster'] == 'Total' ].index
data.drop(indexNames , inplace=True)
indexNames = data[data['Cost/Kg'] == 0 ].index
data.drop(indexNames , inplace=True)
indexNames = data[data['Total Weight (Kg)'] == 0 ].index
data.drop(indexNames , inplace=True)
indexNames = data[data['Kg/Load'] == 0 ].index
data.drop(indexNames , inplace=True)
indexNames = data[data['Kg/Delivery'] == 0 ].index
data.drop(indexNames , inplace=True)
indexNames = data[data['N. of Loads'] == 0 ].index
data.drop(indexNames , inplace=True)



#Dropping the outlier rows with standard deviation
factor = 1
upper_lim = data['Cost/Kg'].mean () + data['Cost/Kg'].std () * factor
lower_lim = data['Cost/Kg'].mean () - data['Cost/Kg'].std () * factor
deleted = data[(data['Cost/Kg'] > upper_lim)]
data = data[(data['Cost/Kg'] < upper_lim) & (data['Cost/Kg'] > lower_lim)]
data2= data.copy()
#new features from stats
data['mean cost/kg cluster'] = data.groupby('Region Cluster')['Cost/Kg'].transform('mean')
data['mean cost/kg region'] = data.groupby('Region')['Cost/Kg'].transform('mean')

#normalizing features
#scaler = MinMaxScaler() 
#num_cols = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))].tolist()
#num_cols.remove('Cost/Kg')

#%%

data = pd.get_dummies(data)

target = np.array(data['Cost/Kg'])

data.drop('Cost/Kg', axis = 1, inplace = True)
features_list = list(data.columns)
features = np.array(data) 

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)

#x_train = features
#y_train = target

y_test_mean1 = x_test[:,4]
y_test_mean2 = x_test[:,5]

#normalizing features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train[:,0:4])
x_train[:,0:4] =(X_train_scaled)
X_test_scaled = scaler.transform(x_test[:,0:4])
x_test[:,0:4] = X_test_scaled



print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)
#%%
rf = RandomForestRegressor(n_estimators = 1000, random_state = 666)
lm = LinearRegression()
lm_lasso = Lasso()
lm_elastic = ElasticNet()
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

#training models
rf.fit(x_train, y_train)
lm.fit(x_train, y_train)
lm_lasso.fit(x_train, y_train)
lm_elastic.fit(x_train, y_train)
print('Training linear SVR')
svr_lin.fit(x_train, y_train)
print('Training poly SVR')
svr_poly.fit(x_train, y_train)
print('Training rbf SVR')
svr_rbf.fit(x_train, y_train)
#%%
#kfold for models

""" 
seed = 666
kfold = model_selection.KFold(n_splits=10, random_state = seed)
scoring = 'neg_mean_absolute_error'
results_rf = model_selection.cross_val_score(rf, features, target, cv=kfold, scoring=scoring)
results_lm = model_selection.cross_val_score(lm, features, target, cv=kfold, scoring=scoring)
results_lm_lasso = model_selection.cross_val_score(lm_lasso, features, target, cv=kfold, scoring=scoring)
results_lm_elastic = model_selection.cross_val_score(lm_elastic, features, target, cv=kfold, scoring=scoring)
results_svr_rbf = model_selection.cross_val_score(svr_rbf, features, target, cv=kfold, scoring=scoring)
results_svr_lin = model_selection.cross_val_score(svr_lin, features, target, cv=kfold, scoring=scoring)
results_svr_poly = model_selection.cross_val_score(svr_poly, features, target, cv=kfold, scoring=scoring)
 """

#%%
#checking on trusted lanes
trusted = pd.read_excel('trusted.xlsx')
trustedy = trusted['Cost/Kg']
trusted.drop(['Cost/Kg','Region', 'Region Cluster','BU','Origin City','Destination Country','Destination City','Total Cost (EUR)'], axis = 1, inplace = True)
trustednp = trusted.to_numpy()
trustedynp = trustedy.to_numpy()
x_test2 = scaler.transform(trustednp)
y_test2 = trustedynp

#%%

x = x_test
y = y_test

y_test_mean1 = x_test[:,4]
y_test_mean2 = x_test[:,5]

#y_pred = rf.predict(x)
#y_pred =y_test_mean1
y_pred =y_test_mean2

errors = abs(y_pred - y)

mape = 100 * (errors / y)
mean_mape = np.mean(mape)

accuracy = 100 - mean_mape

mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

#acc_mae = (1-mae)*100


r2 = r2_score (y, y_pred)
rmse = sqrt(mse)
print('Accuracy:', accuracy, '%.')
print('Root Mean square Error:', rmse)
print('Mean absolute Error:', mae)
print('R2:', r2)

#maekfold = results_rf.mean()

#print ('Mean absolute Error kfold:', maekfold)

#%%

#plotting results

model = rf
visualizer = ResidualsPlot(model)

visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data


visualizer = PredictionError(model)

visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
visualizer.poof()                 # Draw/show/poof the data
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.distplot(data2['Cost/Kg'], label = 'Cost/Kg');

sns.set()
num_cols = data2.columns[data2.dtypes.apply(lambda c: np.issubdtype(c, np.number))].tolist()

#box plot overallqual/saleprice
var = 'Region'
datax = pd.concat([data2['Cost/Kg'], data2[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Cost/Kg", data=datax)
fig.axis(ymin=0, ymax=5);

var = 'Region Cluster'
datax = pd.concat([data2['Cost/Kg'], data2[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Cost/Kg", data=datax)
fig.axis(ymin=0, ymax=5);

#%%
sns.pairplot(data2[num_cols], size = 2)
plt.show();
