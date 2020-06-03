import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from CCIT import CCIT
from CCIT import DataGen
from sklearn.preprocessing import StandardScaler

def ccit():
        
    t1_complete = []
    t2_complete = []
    t1_all_vals = []
    t2_all_vals = []
    rho_vals = []
    for n in [50,100,150,200]:
        print(n)
        t1_all_vals = []
        t2_all_vals = []
        for rho in np.linspace(0,0.95,20):
            print(rho)
            rho_vals.append(rho)
            t_1 = 0
            t_2 = 0
            for i in np.arange(200):
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
                scaler_Z = StandardScaler()
                e_X,e_Y = multivariate_normal(mean=(0,0),cov=[[1,rho],[rho,1]],size=(n)).T
                e_Y = e_Y.reshape(e_Y.shape[0],1)
                e_X = e_X.reshape(e_X.shape[0],1)
                Z = np.random.normal(2,1,n)
                Z = Z.reshape(Z.shape[0],1)
                Z = scaler_Z.fit_transform(Z)
                X = np.sin(15*Z) + e_X
                Y = np.tanh(5*Z) + e_Y
                Y = Y.reshape(Y.shape[0],1)
                Y = scaler_Y.fit_transform(Y)
                X = X.reshape(X.shape[0],1)
                X = scaler_X.fit_transform(X)
                p = CCIT.CCIT(X,Y,Z,max_depths=[11,15,20],num_iter=30,bootstrap=True)
                if rho == 0:
                    if p < 0.05:
                        t_1 += 1
                if rho != 0:
                    if p >= 0.05:
                        t_2 += 1
            if rho == 0:
                t1_all_vals.append(t_1)
            if rho != 0:
                t2_all_vals.append(t_2)
        t1_complete.append(t1_all_vals)
        t2_complete.append(t2_all_vals)
        
    return t1_complete, t2_complete, rho_vals        

ccit_nonlinear_t1, ccit_nonlinear_t2, rho_vals= ccit()