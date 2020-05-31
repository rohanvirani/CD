import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

from CCIT import CCIT
from CCIT import DataGen
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaler_Z = StandardScaler()
X = 3*np.random.normal(0,1,50)
X = X.reshape(X.shape[0],1)
X = scaler_X.fit_transform(X)
Y = 2*np.random.normal(0,1,50)
Y = Y.reshape(Y.shape[0],1)
Y = scaler_Y.fit_transform(Y)
Z = 5*np.random.normal(0,1,50)
Z = Z.reshape(Z.shape[0],1)
Z = scaler_Z.fit_transform(Z)
pvalue = CCIT.CCIT(X,Y,Z,max_depths=[11,15,20],num_iter=30,bootstrap=True)
print(pvalue)