# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:48:32 2022

@author: Fabian
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnchoredText

path = "C:\\Users\\holzme33\\OneDrive - imec\\misc\\project_BEX\\"

year = 2022
# raw data
d = np.loadtxt(path+str(year)+"_raw_data.txt", dtype=str,delimiter='\t')

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
std_err = np.zeros((len(d_1),))
conf_int = np.zeros((len(d_1)))
MAD = np.zeros((len(d_1)))
MAPE = np.zeros((len(d_1)))


# equation (1) from Weyand2006
def eq1_Weyand(P_aer,P_mechmax,k):
    return P_aer+(P_mechmax-P_aer)*np.exp(-k*t)

# confidence limits of correlation coefficient
def CI(r,n,lim):
    # get z for alpha/2 for confidence limit in %
    z = stats.norm.ppf(1-(1-lim/100)/2)
    zr = np.log((1+r)/(1-r))/2      # Fisher transform
    L = zr-(1.96/np.sqrt(n-3))    # lower bounds
    U = zr+(1.96/np.sqrt(n-3))    # upper bounds
    return [(np.exp(2*L)-1)/(np.exp(2*L)+1),(np.exp(2*U)-1)/(np.exp(2*U)+1)]
    


# calculate the model

k_cycle = 0.026
# 2021 data
for i in range(len(d_1)):
    # calculate the model data
    model[i,:] = eq1_Weyand(d_1[i,8],d_1[i,0],k_cycle)
    
    # statistics (evaluate at 5,10,20,30,60,120,300 s)
    t_ev = [5,10,20,30,60,120,300]
    # linear regression of data and prediction
    slope, intercept, r_value, p_value, see = stats.linregress(d_2[i,1:-1],model[i,t_ev])
    r[i] = r_value      # correlation coefficient R
    r2[i] = r_value**2  # R^2
    std_err[i] = see    # standard error of the estimate
    # confidence interval
    lim = 90       # percentile of confidence interval
    conf_int[i] = stats.norm.ppf(1-(1-lim/100)/2)*std_err[i]    # the factor (1.648) is in fact defined for large datasets (normal distribution), but it seems that it is used like this in Sanders2016
    
    # mean absolute deviation and mean absolute percentage error
    MAD[i] = np.mean(abs(model[i,t_ev]-d_2[i,1:-1]))
    MAPE[i] = np.mean((abs(model[i,t_ev]-d_2[i,1:-1]))/d_2[i,1:-1])*100
    
#%% produce and save the figuers
    
for n in range(len(d_1)):
# for n in range(5):
    fig,ax = plt.subplots()
    ax.plot(t[5:351],model[n,5:351],'k-',label='APR model '+str(year)+'/1')
    ax.plot(t_mmp,d_2[n,:],'ro',label='data '+str(year)+'/2')
    ax.set_title(d[n+1,1]+' '+str(year)+'\n'+r"R={:.2f}$\pm${:.2f}".format(r[n],conf_int[n]))
    
    # text box with results of linear regression
    ax.text(200,1000,'R = {:.2f}'.format(r[n])+'\n'+r'R$^2$ = {:.2f}'.format(r2[n])+'\n'+'SEE = {:.2f}'.format(std_err[n])+'\n'
            +'MAD = {:.0f} W'.format(MAD[n])+'\n'+'MAPE = {:.1f} %'.format(MAPE[n]))
    ax.set_xlim([0,370])
    ax.set_ylim([200,1800])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    ax.legend()
    plt.savefig(path+'figures//Athlete'+str(n+1)+'_'+str(year)+'.png')

#%% all athletes together

real_all = np.zeros((7*len(d_1)))
model_all = np.zeros((7*len(d_1)))

for i in range(len(d_1)):
    real_all[i*len(t):(i+1)*len(t)] = d_2[i,1:-1]
    model_all[i*len(t):(i+1)*len(t)] = model[i,t_ev]

slope, intercept, r_value, p_value, std_err_all = stats.linregress(real_all,model_all)
r_all = r_value
lim = 90
conf_int_all = stats.norm.ppf(1-(1-lim/100)/2)*std_err_all
MAD_all = np.mean(abs(real_all-model_all))
MAPE_all = np.mean(abs(real_all-model_all)/real_all)*100

x = np.arange(200,1700)

fig,ax = plt.subplots()
ax.plot(real_all,model_all,'ko')
ax.plot(x,slope*x+intercept,'r')
txt = AnchoredText('R = {:.2f}'.format(r_all)+'\n'+r'R$^2$ = {:.2f}'.format(r_all**2)+'\n'+'SEE = {:.2f}'.format(std_err_all)+'\n'
        +'MAD = {:.0f} W'.format(MAD_all)+'\n'+'MAPE = {:.1f} %'.format(MAPE_all),loc=2)
ax.add_artist(txt)
ax.set_xlabel("Actual Power (W)")
ax.set_ylabel("Predicted Power (W)")
ax.set_title(str(year)+r": R = {:.2f}$\pm${:.2f}".format(r_all,conf_int_all))

#%% correlation coefficient for each duration

for i in range(len(t_ev)):
    data_points = d_2[:,i+1]
    prediction = model[:,t_ev[i]]
    slope, intercept, r_value, p_value, stde = stats.linregress(data_points,prediction)
    MAD_t = np.mean(abs(data_points-prediction))
    MAPE_t = np.mean(abs(data_points-prediction)/data_points)*100
    
    x = np.arange(min(data_points),max(data_points))
    
    fig,ax = plt.subplots()
    ax.plot(data_points,prediction,'ko')
    ax.plot(x,slope*x+intercept,'r')
    txt = AnchoredText('R = {:.2f}'.format(r_value)+'\n'+r'R$^2$ = {:.2f}'.format(r_value**2)+'\n'+'SEE = {:.2f}'.format(stde)+'\n'
            +'MAD = {:.0f} W'.format(MAD_t)+'\n'+'MAPE = {:.1f} %'.format(MAPE_t),loc=2)
    ax.add_artist(txt)
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title(str(year)+', t = '+str(t_ev[i])+' s')
