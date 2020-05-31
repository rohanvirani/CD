import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy.random import multivariate_normal
from Documents.M3R.causal-discovery-for-CI-test.functions import indep,cond_indep,estimate_skeleton

boston = datasets.load_boston()
features = boston.data
features = pd.DataFrame(features)
features = features.drop(columns=[3])
features = np.array(features)
features = pd.DataFrame(features)
features = features.rename({0:'CRIM',1:"ZN",2:"INDUS",3:"NOX",4:"RM",5:"AGE",6:"DIS",7:"RAD",8:"TAX",9:"PTRATIO",10:"B",11:"LSTAT",12:"MEDV"},axis=1)
print(features)

def independence_exp():
        
    HSIC_complete = []
    HSIC_complete_errors = []
    t1_complete = []
    t2_complete = []
    HSIC_all_vals = []
    HSIC_error = []
    t1_all_vals = []
    t2_all_vals = []
    rho_vals = []
    for n in [50,100,150,200]:
        print(n)
        HSIC_all_vals = []
        HSIC_mean = []
        HSIC_error = []
        t1_all_vals = []
        t2_all_vals = []
        for rho in np.linspace(0,0.95,20):
            print(rho)
            rho_vals.append(rho)
            HSIC_vals = []
            t_1 = 0
            t_2 = 0
            for i in np.arange(200):
                X,Y = multivariate_normal(mean=(0,0),cov=[[1,rho],[rho,1]],size=(n)).T
                X = X.reshape(X.shape[0],1)
                Y = Y.reshape(Y.shape[0],1)
                data = np.concatenate((X,Y),axis=1)
                p, HSIC = indep(data,0,1)
                if rho == 0:
                    if p < 0.05:
                        t_1 += 1
                if rho != 0:
                    if p >= 0.05:
                        t_2 += 1
                HSIC_vals.append(HSIC)
            HSIC_mean = np.mean(HSIC_vals)
            HSIC_error.append(np.std(HSIC_vals))
            HSIC_all_vals.append(HSIC_mean)
            if rho == 0:
                t1_all_vals.append(t_1)
            if rho != 0:
                t2_all_vals.append(t_2)
        HSIC_complete_errors.append(HSIC_error)
        HSIC_complete.append(HSIC_all_vals)
        t1_complete.append(t1_all_vals)
        t2_complete.append(t2_all_vals)
        
    return HSIC_complete, HSIC_complete_errors,t1_complete, t2_complete         

HSIC_exp_1, HSIC_error_exp_1, t1_exp_1, t2_exp_1 = independence_exp()
rho_vals=np.linspace(0,0.95,20)
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax1.set_title("HSIC value against rho values for varying n")
ax1.set_xlabel("rho value")
ax1.set_ylabel("HSIC value")
for i in np.arange(4):
    ax1.errorbar(rho_vals,HSIC_exp_1[i],yerr=HSIC_error_exp_1[i],label="n = " + str((i+1)*50))
ax1.legend()
ax2 = fig.add_subplot(1,2,2)
ax2.set_title("Type 2 Errors against rho values for varying n")
ax2.set_xlabel("rho value")
ax2.set_ylabel("Type 2 error count")
for i in np.arange(4):
    ax2.plot(rho_vals[1:20],t2_exp_1[i],label="n = " + str((i+1)*50))
ax2.legend()
print(t1_exp_1)
