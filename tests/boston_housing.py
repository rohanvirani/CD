import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy.random import multivariate_normal
from functions import indep,cond_indep,estimate_skeleton

boston = datasets.load_boston()
features = boston.data
features = pd.DataFrame(features)
features = features.drop(columns=[3])
features = np.array(features)
features = pd.DataFrame(features)
features = features.rename({0:'CRIM',1:"ZN",2:"INDUS",3:"NOX",4:"RM",5:"AGE",6:"DIS",7:"RAD",8:"TAX",9:"PTRATIO",10:"B",11:"LSTAT",12:"MEDV"},axis=1)
estimate_skeleton(cond_indep,features.iloc[:,4:8],0.05)

"""
e_X,e_Y,e_W = multivariate_normal(mean=(0,0,0),cov=[[1,0,0],[0,1,0],[0,0,1]],size=(50)).T
e_Y = e_Y.reshape(e_Y.shape[0],1)
e_X = e_X.reshape(e_X.shape[0],1)
e_W = e_W.reshape(e_W.shape[0],1)
Z = np.random.normal(0,1,50)
Z = Z.reshape(Z.shape[0],1)
Y = np.random.normal(0,1,50)
Y = Y.reshape(Y.shape[0],1)
X = 5*Z + 3*Y + 0.001 + e_X
W = 6*Y + 2*Y + 0.003 + e_W
X = X.reshape(X.shape[0],1)
W = W.reshape(W.shape[0],1)
data = np.concatenate((W,X,Y,Z),axis=1)
estimate_skeleton(cond_indep,data,0.05)
"""