import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy.random import multivariate_normal
from functions import indep,cond_indep,estimate_skeleton
import networkx as nx

boston = datasets.load_boston()
features = boston.data
features = pd.DataFrame(features)
features = features.drop(columns=[3])
features = np.array(features)
features = pd.DataFrame(features)
features = features.rename({0:'CRIM',1:"ZN",2:"INDUS",3:"NOX",4:"RM",5:"AGE",6:"DIS",7:"RAD",8:"TAX",9:"PTRATIO",10:"B",11:"LSTAT"},axis=1)
trial = features.drop(columns={'INDUS','NOX','DIS','RAD','B','LSTAT','CRIM','ZN','PTRATIO'})
estimate_skeleton(cond_indep,trial,0.05)

Z = np.random.normal(2,1,50)
Z = Z.reshape(Z.shape[0],1)
X = 15*Z + 0.0001
Y = 5*Z
Y = Y.reshape(Y.shape[0],1)
X = X.reshape(X.shape[0],1)
data = np.concatenate((X,Y,Z),axis=1)
a, b = estimate_skeleton(cond_indep, data, 0.05)
print(b)
nx.draw_networkx(a)
plt.show()

