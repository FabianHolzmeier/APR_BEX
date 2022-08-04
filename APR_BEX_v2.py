# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:48:32 2022

@author: Fabian
"""
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\Fabian\\OneDrive - imec\\misc\\project_BEX\\"

# raw data
d = np.loadtxt(path+"2021_raw_data.txt", dtype=str,delimiter='\t')

# first period
d_1 = d[1:,5:14].astype(float)
# second period 2021
d_2 = d[1:,14:].astype(float)

# time (s) of MMP
t_mmp = np.array([2,5,10,20,30,60,120,300,360])

# time (s) at which the model should be evaluated
t = np.arange(0,361)

# initialize arrays with result of model
model = np.zeros((len(d_1),len(t)))
r = np.zeros((len(d_1)))
r2 = np.zeros((len(d_1),))
see = np.zeros((len(d_1),))


# equation (1) from Weyand2006
def eq1_Weyand(P_aer,P_mechmax,k):
    return P_aer+(P_mechmax-P_aer)*np.exp(-k*t)

# calculate the model

k_cycle = 0.026
# 2021 data
for i in range(len(d_1)):
    # calculate the model data
    model[i,:] = eq1_Weyand(d_1[i,8],d_1[i,0],k_cycle)
    
    # correlation coefficients (evaluate at 5,10,20,30,60,120,300 s)
    r[i] = np.corrcoef(d_2[i,1:-1],model[i,[5,10,20,30,60,120,300]])[0,1]
    r2[i] = r[i]**2   
    
    # standard error of the estimate
    sigma = np.sqrt(np.sum((d_2[i,1:-1]-model[i,[5,10,20,30,60,120,300]])**2)/(7-1))
    see[i] = sigma/np.sqrt(7)

    
    
#%% produce and save the figuers
    
# for n in range(len(d_1)):
for n in range(5):
    fig,ax = plt.subplots()
    ax.plot(t[5:351],model[n,5:351],'k-',label='APR model 2021/1')
    ax.plot(t[5:351],model[n,5:351]+see[n],'k:',label='+ SEE {:.1f}'.format(see[n]))
    ax.plot(t[5:351],model[n,5:351]-see[n],'k:')
    ax.plot(t_mmp,d_2[n,:],'ro',label='data 2021/2')
    ax.set_title(d[n+1,1]+r", R$^2$={:.2f}".format(r2[n]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    ax.legend()
    # plt.savefig(path+'figures//Athlete'+str(n+1)+'_2021.png')

#%% all athletes together

real_all = np.zeros((7*len(d_1)))
model_all = np.zeros((7*len(d_1)))

for i in range(len(d_1)):
    real_all[i*7:(i+1)*7] = d_2[i,1:-1]
    model_all[i*7:(i+1)*7] = model[i,[5,10,20,30,60,120,300]]

r_all = np.corrcoef(real_all,model_all)[0][1]
r2_all = r_all**2

fig,ax = plt.subplots()
ax.plot(real_all,model_all,'ko')
ax.set_xlabel("Actual Power (W)")
ax.set_ylabel("Predicted Power (W)")
ax.set_title(r"2021: R$^2$ = {:.2f}".format(r2_all))
