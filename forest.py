import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy.sparse import data
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV



def m(x):
  # x is a numpy array with 5 components
  return 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*(x[:,2]-0.05)**2 + 10*x[:,3] + 5*x[:,4]


def small_noise_Y(x): 
  noise = np.random.normal(0,0.1) 
  Y = m(x) + noise
  return Y


def large_noise_Y(x): 
  noise = np.random.normal(0,1) 
  Y = m(x) + noise
  return Y




ITER = 10
p = 5 # dimension of x
data_size= 1000
M = 500
loss_table = np.zeros(ITER)
data_table = np.zeros(ITER)
for i in range(ITER):
    data_size = 1000 * 2**i
    regressor = RandomForestRegressor(M,max_leaf_nodes = int(data_size**0.7))
    X_train = np.random.uniform(low=0, high=1, size=(data_size,p))
    Y_train = m(X_train)
    X_test = np.random.uniform(low=0, high=1, size=(1000,p))
    Y_test = m(X_test)
    regressor.fit(X_train, Y_train)
    Y_predicted = regressor.predict(X_test)
    loss = sklearn.metrics.mean_squared_error(Y_predicted, Y_test)
    data_table[i] = data_size
    loss_table[i] = loss
plt.plot(data_table,loss_table)




ITER = 10
p = 5 # dimension of x
data_size= 1000
M = 500
loss_table = np.zeros(ITER)
data_table = np.zeros(ITER)
for i in range(ITER):
    data_size = 1000 * 2**i
    regressor = RandomForestRegressor(M,max_leaf_nodes = int(data_size**0.7))
    X_train = np.random.uniform(low=0, high=1, size=(data_size,p))
    Y_train = small_noise_Y(X_train)
    X_test = np.random.uniform(low=0, high=1, size=(1000,p))
    Y_test = m(X_test)
    regressor.fit(X_train, Y_train)
    Y_predicted = regressor.predict(X_test)
    loss = sklearn.metrics.mean_squared_error(Y_predicted, Y_test)
    data_table[i] = data_size
    loss_table[i] = loss
plt.plot(data_table,loss_table)




ITER = 10
p = 5 # dimension of x
data_size= 1000
M = 10
loss_table = np.zeros(ITER)
data_table = np.zeros(ITER)
for i in range(ITER):
    data_size = 1000 * 2**i
    regressor = RandomForestRegressor(M,max_leaf_nodes = int(data_size**0.7))
    X_train = np.random.uniform(low=0, high=1, size=(data_size,p))
    Y_train = large_noise_Y(X_train)
    X_test = np.random.uniform(low=0, high=1, size=(1000,p))
    Y_test = m(X_test)
    regressor.fit(X_train, Y_train)
    Y_predicted = regressor.predict(X_test)
    loss = sklearn.metrics.mean_squared_error(Y_predicted, Y_test)
    data_table[i] = data_size
    loss_table[i] = loss
plt.plot(data_table,loss_table)



ITER = 6
p = 5 # dimension of x
data_size= 1000
M = 10
loss_table = np.zeros(ITER)
data_table = np.zeros(ITER)
for i in range(ITER):
    data_size = 1000 * 2**i
    regressor = RandomForestRegressor(M,max_leaf_nodes = int(data_size**0.7))
    X_train = np.random.uniform(low=0, high=1, size=(data_size,p))
    Y_train = m(X_train)
    X_test = np.random.uniform(low=0, high=1, size=(1000,p))
    Y_test = m(X_test)
    regressor.fit(X_train, Y_train)
    Y_predicted = regressor.predict(X_test)
    loss = sklearn.metrics.mean_squared_error(Y_predicted, Y_test)
    data_table[i] = data_size
    loss_table[i] = loss
plt.plot(data_table,loss_table)



ITER = 10
p = 5 # dimension of x
data_size= 1000
M = 10
loss_table = np.zeros(ITER)
data_table = np.zeros(ITER)
for i in range(ITER):
    data_size = 1000
    regressor = RandomForestRegressor(M,max_leaf_nodes = int(data_size**0.7))
    X_train = np.random.uniform(low=0, high=1, size=(data_size,p))
    Y_train = m(X_train)
    X_test = np.random.uniform(low=0, high=1, size=(1000,p))
    Y_test = m(X_test)
    regressor.fit(X_train, Y_train)
    Y_predicted = regressor.predict(X_test)
    loss = sklearn.metrics.mean_squared_error(Y_predicted, Y_test)
    data_table[i] = M
    loss_table[i] = loss
    M = 2 * M
plt.plot(data_table,loss_table)