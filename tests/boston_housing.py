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
features = features.rename({0:'CRIM',1:"ZN",2:"INDUS",3:"NOX",4:"RM",5:"AGE",6:"DIS",7:"RAD",8:"TAX",9:"PTRATIO",10:"B",11:"LSTAT"},axis=1)
trial = features.drop(columns={'INDUS','NOX','DIS','RAD','B','LSTAT'})
estimate_skeleton(cond_indep,trial,0.05)

