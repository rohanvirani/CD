import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gcm_linear_t1 = pd.read_csv('~/Documents/M3R/CD/gcm_linear_t1.csv')
gcm_linear_t2 = pd.read_csv('~/Documents/M3R/CD/gcm_linear_t2.csv')

kci_linear_t1 = pd.read_csv('~/Documents/M3R/CD/kci_linear_t1.csv')
kci_linear_t2 = pd.read_csv('~/Documents/M3R/CD/kci_linear_t2.csv')

resit_linear_t1 = pd.read_csv('~/Documents/M3R/CD/tests/resit_linear_t1.csv')
resit_linear_t2 = pd.read_csv('~/Documents/M3R/CD/tests/resit_linear_t2.csv')

gcm_1 = np.array(gcm_linear_t1.iloc[:,1])
gcm_2 = np.array(gcm_linear_t2.iloc[:,1])
kci_1 = np.array(kci_linear_t1.iloc[:,1])
kci_2 = np.array(kci_linear_t2.iloc[:,1])
res_1 = np.array(resit_linear_t1.iloc[:,1:])
res_2 = np.array(resit_linear_t2.iloc[:,1:])

gcm_linear_t1 = gcm_linear_t1.drop(columns='Unnamed: 0')
kci_linear_t1 = kci_linear_t1.drop(columns='Unnamed: 0')
resit_linear_t1 = resit_linear_t1.drop(columns='Unnamed: 0')

gcm_linear_t1 = gcm_linear_t1.rename(columns={'t1':'GCM'})
kci_linear_t1 = kci_linear_t1.rename(columns={'t1':'KCI'})
resit_linear_t1 = resit_linear_t1.rename(columns={'0':'RESIT'})
t1_data = pd.concat([gcm_linear_t1,kci_linear_t1,resit_linear_t1],axis=1)
t1_data = t1_data.rename(index={0:'50',1:'100',2:'150',3:'200'})
ax = plt.axes()
sns.heatmap(t1_data, ax = ax)
ax.set_title('Heatmap of Type 1 error % for different tests')
plt.show()

rho_vals = np.linspace(0,0.95,20)
fig = plt.figure(figsize=(15,12))
ax2 = fig.add_subplot(2,2,1)
ax2.set_title("Type 2 Error % against rho values for n=50")
ax2.set_xlabel("rho value")
ax2.set_ylabel("Type 2 error %")
ax2.plot(rho_vals[1:20],gcm_2[:19],label="GCM")
ax2.plot(rho_vals[1:20],kci_2[:19],label="KCI")
ax2.plot(rho_vals[1:20],res_2[0],label="RESIT")
ax2.legend()

ax3 = fig.add_subplot(2,2,2)
ax3.set_title("Type 2 Error % against rho values for n=100")
ax3.set_xlabel("rho value")
ax3.set_ylabel("Type 2 error %")
ax3.plot(rho_vals[1:20],gcm_2[19:38],label="GCM")
ax3.plot(rho_vals[1:20],kci_2[19:38],label="KCI")
ax3.plot(rho_vals[1:20],res_2[1],label="RESIT")
ax3.legend()

ax4 = fig.add_subplot(2,2,3)
ax4.set_title("Type 2 Error % against rho values for n=150")
ax4.set_xlabel("rho value")
ax4.set_ylabel("Type 2 error %")
ax4.plot(rho_vals[1:20],gcm_2[38:57],label="GCM")
ax4.plot(rho_vals[1:20],kci_2[38:57],label="KCI")
ax4.plot(rho_vals[1:20],res_2[2],label="RESIT")
ax4.legend()

ax5 = fig.add_subplot(2,2,4)
ax5.set_title("Type 2 Error % against rho values for n=200")
ax5.set_xlabel("rho value")
ax5.set_ylabel("Type 2 error %")
ax5.plot(rho_vals[1:20],gcm_2[57:],label="GCM")
ax5.plot(rho_vals[1:20],kci_2[57:],label="KCI")
ax5.plot(rho_vals[1:20],res_2[3],label="RESIT")
ax5.legend()

plt.show()
