import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from numpy import shape, fill_diagonal, zeros, mean, sqrt,identity,dot,diag
from numpy.random import permutation, randn
from scipy.spatial.distance import squareform, pdist
from numpy import exp, shape, reshape, sqrt, median
from itertools import combinations, permutations
import networkx as nx
from numpy.random import multivariate_normal
from numpy.random import multivariate_normal
from pygam import GAM, s, f, te

boston = datasets.load_boston()
features = boston.data
features = pd.DataFrame(features)
features = features.drop(columns=[3])
features = np.array(features)
features = pd.DataFrame(features)
features = features.rename({0:'CRIM',1:"ZN",2:"INDUS",3:"NOX",4:"RM",5:"AGE",6:"DIS",7:"RAD",8:"TAX",9:"PTRATIO",10:"B",11:"LSTAT",12:"MEDV"},axis=1)
print(features)

def indep(data_matrix,i,j):
    data_matrix = np.array(data_matrix)
    X = data_matrix[:,i]
    X = X.reshape(X.shape[0],1)
    Y = data_matrix[:,j]
    Y = Y.reshape(Y.shape[0],1)


    def get_sigma(X):
        n=shape(X)[0]
        if n>1000:
            X=X[permutation(n)[:1000],:]
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=median_dist/sqrt(2.)
        return sigma
    
    def kernel(X,sigma):
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = exp(-0.5 * (sq_dists) / sigma ** 2)

        return K


    sigmax = get_sigma(X)
    sigmay = get_sigma(Y)
    Kx=kernel(X,sigmax)
    Ky=kernel(Y,sigmay)

    def U_statistic(Kx,Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx,0.)
        fill_diagonal(Ky,0.)
        K = np.dot(Kx,Ky)
        first_term = np.trace(K)/float(m*(m-3.))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term
    
    num_shuffles=1000
    ny=shape(Y)[0]
    test_statistic = U_statistic(Kx,Ky)
    null_samples=zeros(num_shuffles)
    for jj in range(num_shuffles):
        pp = permutation(ny)
        Kpp = Ky[pp,:][:,pp]
        null_samples[jj]=U_statistic(Kx,Kpp)

    pvalue = ( 1 + sum( null_samples > test_statistic ) ) / float( 1 + 1000)
  
    return pvalue, test_statistic

def cond_indep(data_matrix,i,j,k):
    data_matrix=np.array(data_matrix)
    X = data_matrix[:,i]
    Y = data_matrix[:,j]
    k = list(k)
    Z = data_matrix[:,k]

    GPR_1 = GaussianProcessRegressor()
    GPR_1.fit(np.array(Z),np.array(X))
    predictions_gpr_1 = GPR_1.predict(np.array(Z))
    res_gpr_1 = predictions_gpr_1 - np.array(X)

    GPR_2 = GaussianProcessRegressor()
    GPR_2.fit(np.array(Z),np.array(Y))
    predictions_gpr_2 = GPR_2.predict(np.array(Z))
    res_gpr_2 = predictions_gpr_2 - np.array(Y)
    
    res_x = res_gpr_1.reshape(res_gpr_1.shape[0],1)
    res_y = res_gpr_2.reshape(res_gpr_2.shape[0],1)

    def get_sigma(X):
        n=shape(X)[0]
        if n>1000:
            X=X[permutation(n)[:1000],:]
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=median_dist/sqrt(2.)
        return sigma
    
    def kernel(X,sigma):
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = exp(-0.5 * (sq_dists) / sigma ** 2)

        return K


    sigmax = get_sigma(res_x)
    sigmay = get_sigma(res_y)
    Kx=kernel(res_x,sigmax)
    Ky=kernel(res_y,sigmay)

    def U_statistic(Kx,Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx,0.)
        fill_diagonal(Ky,0.)
        K = np.dot(Kx,Ky)
        first_term = np.trace(K)/float(m*(m-3.))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term
    
    num_shuffles=1000
    ny=shape(res_y)[0]
    test_statistic = U_statistic(Kx,Ky)
    null_samples=zeros(num_shuffles)
    for jj in range(num_shuffles):
        pp = permutation(ny)
        Kpp = Ky[pp,:][:,pp]
        null_samples[jj]=U_statistic(Kx,Kpp)

    pvalue = ( 1 + sum( null_samples > test_statistic ) ) / float( 1 + 1000)
  
    return pvalue, res_gpr_1, res_gpr_2, test_statistic

def cond_indep_ensemble(data_matrix,i,j,k):
    X = data_matrix[:,i]
    Y = data_matrix[:,j]
    k = list(k)
    Z = data_matrix[:,k]

    GPR_1 = GaussianProcessRegressor()
    GPR_1.fit(np.array(Z),np.array(X))
    predictions_gpr_1 = GPR_1.predict(np.array(Z).reshape(np.shape(Z)[0],1))
    res_gpr_1 = predictions_gpr_1 - np.array(X)

    GPR_2 = GaussianProcessRegressor()
    GPR_2.fit(np.array(Z),np.array(Y))
    predictions_gpr_2 = GPR_2.predict(np.array(Z).reshape(np.shape(Z)[0],1))
    res_gpr_2 = predictions_gpr_2 - np.array(Y)
    
    """
    MLP_1 = MLPRegressor(hidden_layer_sizes=(5,5),shuffle=True,max_iter=500,alpha=0.1)
    MLP_1.fit(np.array(Z),np.array(X))
    predictions_mlp_1 = MLP_1.predict(np.array(Z).reshape(np.shape(Z)[0],1))
    res_mlp_1 = predictions_mlp_1 - np.array(X).reshape(np.shape(X)[0])
    
    MLP_2 = MLPRegressor(hidden_layer_sizes=(5,5),shuffle=True,max_iter=500,alpha=0.1)
    MLP_2.fit(np.array(Z),np.array(Y))
    predictions_mlp_2 = MLP_2.predict(np.array(Z).reshape(np.shape(Z)[0],1))
    res_mlp_2 = predictions_mlp_2 - np.array(Y).reshape(np.shape(Y)[0])
    """
    GAM_1 = GAM(n_splines=15).fit(np.array(Z),np.array(X))
    predictions_gam_1 = GAM_1.predict(np.array(Z))
    res_gam_1 = predictions_gam_1 - np.array(X).reshape(np.shape(X)[0])
    
    GAM_2 = GAM(n_splines=15).fit(np.array(Z),np.array(Y))
    predictions_gam_2 = GAM_2.predict(np.array(Z))
    res_gam_2 = predictions_gam_2 - np.array(Y).reshape(np.shape(Y)[0])
    
    GAM_3 = GAM(n_splines=20).fit(np.array(Z),np.array(X))
    predictions_gam_3 = GAM_3.predict(np.array(Z))
    res_gam_3 = predictions_gam_3 - np.array(X).reshape(np.shape(X)[0])
    
    GAM_4 = GAM(n_splines=20).fit(np.array(Z),np.array(Y))
    predictions_gam_4 = GAM_4.predict(np.array(Z))
    res_gam_4 = predictions_gam_4 - np.array(Y).reshape(np.shape(Y)[0])
    
    GAM_5 = GAM(n_splines=50).fit(np.array(Z),np.array(X))
    predictions_gam_5 = GAM_5.predict(np.array(Z))
    res_gam_5 = predictions_gam_5 - np.array(X).reshape(np.shape(X)[0])
    
    GAM_6 = GAM(n_splines=50).fit(np.array(Z),np.array(Y))
    predictions_gam_6 = GAM_6.predict(np.array(Z))
    res_gam_6 = predictions_gam_6 - np.array(Y).reshape(np.shape(Y)[0])
    
    res_x = (1/4)*(res_gpr_1.reshape(res_gpr_1.shape[0],1) + res_gam_1.reshape(res_gam_1.shape[0],1) + res_gam_3.reshape(res_gam_3.shape[0],1)+res_gam_5.reshape(res_gam_5.shape[0],1))
    res_y = (1/4)*(res_gpr_2.reshape(res_gpr_2.shape[0],1) +res_gam_2.reshape(res_gam_2.shape[0],1) + res_gam_4.reshape(res_gam_4.shape[0],1) + res_gam_6.reshape(res_gam_6.shape[0],1))

    def get_sigma(X):
        n=shape(X)[0]
        if n>1000:
            X=X[permutation(n)[:1000],:]
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=median_dist/sqrt(2.)
        return sigma
    
    def kernel(X,sigma):
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = exp(-0.5 * (sq_dists) / sigma ** 2)

        return K


    sigmax = get_sigma(res_x)
    sigmay = get_sigma(res_y)
    Kx=kernel(res_x,sigmax)
    Ky=kernel(res_y,sigmay)

    def U_statistic(Kx,Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx,0.)
        fill_diagonal(Ky,0.)
        K = np.dot(Kx,Ky)
        first_term = np.trace(K)/float(m*(m-3.))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term
    
    num_shuffles=1000
    ny=shape(res_y)[0]
    test_statistic = U_statistic(Kx,Ky)
    null_samples=zeros(num_shuffles)
    for jj in range(num_shuffles):
        pp = permutation(ny)
        Kpp = Ky[pp,:][:,pp]
        null_samples[jj]=U_statistic(Kx,Kpp)

    pvalue = ( 1 + sum( null_samples > test_statistic ) ) / float( 1 + 1000)
  
    return pvalue, res_x, res_y, test_statistic

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def estimate_skeleton(indep_test_func, data_matrix, alpha, **kwargs):
    """Estimate a skeleton graph from the statistis information.
    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"
    node_ids = range(data_matrix.shape[1])
    g = _create_complete_graph(node_ids)
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    

    l = 0
    completed_z_idx = 0
    completed_xy_idx = 0
    while True:
        cont = False
        remove_edges = []
        perm_iteration_list = list(permutations(node_ids,2))
        length_iteration_list = len(perm_iteration_list)
        for ij in np.arange(completed_xy_idx, length_iteration_list):
            (i,j) = perm_iteration_list[ij]
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                if len(adj_i) < l:
                    continue
                cc = list(combinations(adj_i, l))
                length_cc = len(cc)
                
                
                for kk in np.arange(completed_z_idx, length_cc):
                    print(kk)
                    k = cc[kk]
                    print(k)
                    if l == 0: 
                        
                        p_val, test_statistic = indep(data_matrix,i,j)
                        
                    else: # conditional independence testing
                        
                        p_val, res_gpr_1, res_gpr_2, test_statistic = cond_indep(data_matrix,i,j,k)
                        
                        
                    completed_z_idx = kk + 1
                    if p_val > alpha:
                        if g.has_edge(i, j):
                           
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                completed_z_idx = 0
                completed_xy_idx = ij + 1
                cont = True
        l += 1
        completed_xy_idx = 0
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return (nx.draw_networkx(g), sep_set)

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
