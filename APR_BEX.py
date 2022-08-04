# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:48:32 2022

@author: Fabian
"""
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\Fabian\\OneDrive - imec\\misc\\project_BEX\\"

# raw data 2021
d2021 = np.loadtxt(path+"2021_raw_data.txt", dtype=str,delimiter='\t')
# raw data 2022
d2022 = np.loadtxt(path+"2022_raw_data.txt", dtype=str,delimiter='\t')

# first period 2021
d2021_1 = d2021[1:,5:14].astype(float)
# second period 2021
d2021_2 = d2021[1:,14:].astype(float)
# first period 2022
d2022_1 = d2022[1:,5:14].astype(float)
# second period 2022
d2022_2 = d2022[1:,14:].astype(float)

# time (s) of MMP
t_mmp = np.array([2,5,10,20,30,60,120,300,360])


# time (s) at which the model should be evaluated
t = np.arange(0,361)

# initialize arrays with result of model
model_21 = np.zeros((len(d2021_1),len(t)))
r2_21 = np.zeros((len(d2021_1),))
model_22 = np.zeros((len(d2022_1),len(t)))
r2_22 = np.zeros((len(d2022_1),))

# equation (1) from Weyand2006
def eq1_Weyand(P_aer,P_mechmax,k):
    return P_aer+(P_mechmax-P_aer)*np.exp(-k*t)

# calculate the model

k_cycle = 0.026
# 2021 data
for i in range(len(d2021_1)):
    model_21[i,:] = eq1_Weyand(d2021_1[i,8],d2021_1[i,0],k_cycle)
    # calculate R_square (evaluate at 5,10,20,30,60,120,300,360 s)
    r2_21[i] = np.corrcoef(d2021_2[i,1:],model_21[i,[5,10,20,30,60,120,300,360]])[0,1]**2   # to be checked, R**2 seems quite high
    
# 2022 data  
for i in range(len(d2022_1)):
    model_22[i,:] = eq1_Weyand(d2022_1[i,8],d2022_1[i,0],k_cycle)
    # calculate R_square (evaluate at 5,10,20,30,60,120,300,360 s)
    r2_22[i] = np.corrcoef(d2022_2[i,1:],model_22[i,[5,10,20,30,60,120,300,360]])[0,1]**2
    
#%% produce and save the figuers
    
for n in range(len(d2021_1)):
    fig,ax = plt.subplots()
    ax.plot(t[5:351],model_21[n,5:351],'k-',label='APR model 2021/1')
    ax.plot(t_mmp,d2021_2[n,:],'ro',label='data 2021/2')
    ax.set_title(d2021[n+1,1]+r", R$^2$={:.2f}".format(r2_21[n]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    ax.legend()
    plt.savefig(path+'figures//Athlete'+str(n+1)+'_2021.png')


for n in range(len(d2022_1)):
    fig,ax = plt.subplots()
    ax.plot(t[5:351],model_22[n,5:351],'k-',label='APR model 2022/1')
    ax.plot(t_mmp,d2022_2[n,:],'ro',label='data 2022/2')
    ax.set_title(d2022[n+1,1]+r", R$^2$={:.2f}".format(r2_22[n]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    ax.legend()
    plt.savefig(path+'figures//Athlete'+str(n+1)+'_2022.png')