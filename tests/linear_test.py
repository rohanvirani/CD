import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from functions import indep,cond_indep,estimate_skeleton

def independence_exp():
        
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
                e_X,e_Y = multivariate_normal(mean=(0,0),cov=[[1,rho],[rho,1]],size=(n)).T
                e_Y = e_Y.reshape(e_Y.shape[0],1)
                e_X = e_X.reshape(e_X.shape[0],1)
                Z = np.random.normal(2,1,n)
                Z = Z.reshape(Z.shape[0],1)
                X = 15*Z + 0.0001+ e_X
                Y = 5*Z + 0.004 + e_Y
                Y = Y.reshape(Y.shape[0],1)
                X = X.reshape(X.shape[0],1)
                data = np.concatenate((X,Y,Z),axis=1)
                p, _, _ ,_ = cond_indep(data,0,1,[2])
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

t1_linear, t2_linear, rho_vals= independence_exp()

fig = plt.figure(figsize=(15,5))
ax2 = fig.add_subplot(1,2,2)
ax2.set_title("Type 2 Errors against rho values for varying n")
ax2.set_xlabel("rho value")
ax2.set_ylabel("Type 2 error count")
for i in np.arange(4):
    ax2.plot(rho_vals[1:20],t2_linear[i],label="n = " + str((i+1)*50))
ax2.legend()

print(t1_linear)
