# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:48:32 2022

@author: Fabian
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnchoredText

path = "C:\\Users\\Fabian\\OneDrive - imec\\misc\\project_BEX\\"

year = 2021
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
    L = zr-(z/np.sqrt(n-3))    # lower bounds
    U = zr+(z/np.sqrt(n-3))    # upper bounds
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
    # std_err[i] = see    # standard error of the estimated SLOPE 
    
    # standard error of the estimate in Watt
    std_err[i] = np.sqrt(np.sum((model[i,t_ev]-d_2[i,1:-1])**2)/len(t_ev))
    
    # confidence interval
    lim = 90       # percentile of confidence interval
    # calculate uper and lower confidence limits for R 
    # (in my opinion, one should do this, but Sanders et al. did it differently, so asymmetric confidence intervals are not used at the moment)
    ci = CI(r[i],len(t_ev),lim)     
    # I think that Sanders et al. (ref. 3) used this formula for their error on R, but the asymmetric error (ci):
    # assuming that they had 16 interval times where they compared real data to the model, I get more or less the same error they show)
    conf_int[i] = stats.norm.ppf(1-(1-lim/100)/2)*np.sqrt(1-r2[i])/np.sqrt(len(t_ev)-2)   # the factor stast.norm.ppf (1.648 for 90%) is in fact defined for large datasets (normal distribution), but it seems that it is used like this in Sanders2016
    # print(r[i],ci-r[i],conf_int[i])
    
    # mean absolute deviation and mean absolute percentage error
    MAD[i] = np.mean(abs(model[i,t_ev]-d_2[i,1:-1]))
    MAPE[i] = np.mean((abs(model[i,t_ev]-d_2[i,1:-1]))/d_2[i,1:-1])*100
    
#%% produce and save figures for each athlete
    
for n in range(len(d_1)):
# for n in range(5):
    fig,ax = plt.subplots()
    ax.plot(t[5:351],model[n,5:351],'k-',label='APR model '+str(year)+'/1')
    ax.plot(t_mmp,d_2[n,:],'ro',label='data '+str(year)+'/2')
    ax.set_title(d[n+1,1]+' '+str(year)+'\n'+r"R={:.2f}$\pm${:.2f}".format(r[n],conf_int[n]))
    
    # text box with results of linear regression
    txt = AnchoredText('R = {:.2f}'.format(r[n])+'\n'+r'R$^2$ = {:.2f}'.format(r2[n])+'\n'+'SEE = {:.1f} W'.format(std_err[n])+'\n'
            +'MAD = {:.1f} W'.format(MAD[n])+'\n'+'MAPE = {:.1f} %'.format(MAPE[n]),loc='center')
    ax.add_artist(txt)
    ax.set_xlim([0,370])
    ax.set_ylim([200,1800])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (w)")
    ax.legend()
    plt.savefig(path+'figures//Athlete'+str(n+1)+'_'+str(year)+'.png')

#%% overview of R, MAPE

fig,ax = plt.subplots()
ax.errorbar(np.arange(1,len(d_1)+1),r,yerr = conf_int,color='k',linestyle='',marker='o',capsize=5)
lns1 = ax.plot(np.arange(1,len(d_1)+1),r,'ko',label="R")
lns2 = ax.plot([1,len(d_1)+1],[np.mean(r),np.mean(r)],'r-',label="mean of R")
ax.set_xlabel("Athlete No.")
ax.set_ylabel("Correlation Coefficient R")

ax.set_ylim([0.2,1.6])

ax2 = ax.twinx()
lns3 = ax2.plot(np.arange(1,len(d_1)+1),MAPE,'b^',label="MAPE")
lns4 = ax2.plot([1,len(d_1)+1],[np.mean(MAPE),np.mean(MAPE)],'b--',label="mean of MAPE")
ax2.set_ylabel("MAPE (%)")
ax2.set_ylim([0,50])

# write mean values and simple standard deviation for R and MAPE in the title
ax.set_title(str(year)+r": R = {:.2f}$\pm${:.2f}, MAPE = {:.1f}$\pm${:.1f} %".format(np.mean(r),np.std(r),np.mean(MAPE),np.std(MAPE)))

lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig(path+'figures//overview_R_MAPE_'+str(year)+'.png')

#%% all athletes together

real_all = np.zeros((7*len(d_1)))
model_all = np.zeros((7*len(d_1)))

for i in range(len(d_1)):
    real_all[i*len(t_ev):(i+1)*len(t_ev)] = d_2[i,1:-1]
    model_all[i*len(t_ev):(i+1)*len(t_ev)] = model[i,t_ev]

slope, intercept, r_value, p_value, stderr = stats.linregress(real_all,model_all)
r_all = r_value
std_err_all = np.sqrt(np.sum((model_all-real_all)**2)/len(model_all))
lim = 90
conf_int_all = stats.norm.ppf(1-(1-lim/100)/2)*np.sqrt(1-r_all**2)/np.sqrt(len(model_all)-2)
MAD_all = np.mean(abs(real_all-model_all))
MAPE_all = np.mean(abs(real_all-model_all)/real_all)*100

x = np.arange(200,1700)

fig,ax = plt.subplots()
ax.plot(real_all,model_all,'ko')
ax.plot(x,slope*x+intercept,'r')
txt = AnchoredText('R = {:.2f}'.format(r_all)+'\n'+r'R$^2$ = {:.2f}'.format(r_all**2)+'\n'+'SEE = {:.1f} W'.format(std_err_all)+'\n'
        +'MAD = {:.1f} W'.format(MAD_all)+'\n'+'MAPE = {:.1f} %'.format(MAPE_all),loc=2)
ax.add_artist(txt)
ax.set_xlabel("Actual Power (W)")
ax.set_ylabel("Predicted Power (W)")
ax.set_title(str(year)+r": R = {:.2f}$\pm${:.2f}".format(r_all,conf_int_all))
plt.savefig(path+'figures//all_athletes_'+str(year)+'.png')

#%% filter on athletes

filt = 'sprinter'       # filter options: male, female, sprinter, non-sprinter

if filt=='male':
    idx = np.where(d[:,3]=='male')[0]
    title = 'all male athletes '
elif filt == 'female':
    idx = np.where(d[:,3]=='female')[0]
    title = 'all female athletes '
elif filt == 'sprinter':
    idx = np.where(d_1[:,0]/d_1[:,8]>2.73)[0]   # sprinters have P_mechmax/P_aer > 2.73
    idx = idx+1
    title = 'sprinters '
elif filt == 'non-sprinter':
    idx = np.where(d_1[:,0]/d_1[:,8]<2.73)[0]   
    idx = idx+1
    title = 'non-sprinters '
else:
    print("option not available")
    
real_filt = np.zeros((7*len(idx)))
model_filt = np.zeros((7*len(idx)))

for i in range(len(idx)):
    real_filt[i*len(t_ev):(i+1)*len(t_ev)] = d_2[idx[i]-1,1:-1]
    model_filt[i*len(t_ev):(i+1)*len(t_ev)] = model[idx[i]-1,t_ev]

slope, intercept, r_value, p_value, stderr = stats.linregress(real_filt,model_filt)
r_filt = r_value
std_err_filt = np.sqrt(np.sum((model_filt-real_filt)**2)/len(model_filt))
lim = 90
conf_int_filt = stats.norm.ppf(1-(1-lim/100)/2)*np.sqrt(1-r_filt**2)/np.sqrt(len(model_filt)-2)
MAD_filt = np.mean(abs(real_filt-model_filt))
MAPE_filt = np.mean(abs(real_filt-model_filt)/real_filt)*100

x = np.arange(200,1700)

fig,ax = plt.subplots()
ax.plot(real_filt,model_filt,'ko')
ax.plot(x,slope*x+intercept,'r')
txt = AnchoredText('R = {:.2f}'.format(r_filt)+'\n'+r'R$^2$ = {:.2f}'.format(r_filt**2)+'\n'+'SEE = {:.1f} W'.format(std_err_filt)+'\n'
        +'MAD = {:.1f} W'.format(MAD_filt)+'\n'+'MAPE = {:.1f} %'.format(MAPE_filt),loc=2)
ax.add_artist(txt)
ax.set_xlabel("Actual Power (W)")
ax.set_ylabel("Predicted Power (W)")
ax.set_title(title+str(year)+'\n'+r"R = {:.2f}$\pm${:.2f}".format(r_filt,conf_int_filt))
plt.savefig(path+'figures//filter_'+filt+'_'+str(year)+'.png')
#%% correlation coefficient for each duration

for i in range(len(t_ev)):
    data_points = d_2[:,i+1]
    prediction = model[:,t_ev[i]]
    slope, intercept, r_value, p_value, stderr = stats.linregress(data_points,prediction)
    stde = np.sqrt(np.sum((prediction-data_points)**2)/len(data_points))
    MAD_t = np.mean(abs(data_points-prediction))
    MAPE_t = np.mean(abs(data_points-prediction)/data_points)*100
    lim = 90
    conf_int_t = stats.norm.ppf(1-(1-lim/100)/2)*np.sqrt(1-r_value**2)/np.sqrt(len(data_points)-2)
    
    x = np.arange(min(data_points),max(data_points))
    
    fig,ax = plt.subplots()
    ax.plot(data_points,prediction,'ko')
    ax.plot(x,slope*x+intercept,'r')
    txt = AnchoredText(r'R = {:.2f} $\pm$ {:.2f}'.format(r_value,conf_int_t)+'\n'+r'R$^2$ = {:.2f}'.format(r_value**2)+'\n'+'SEE = {:.1f} W'.format(stde)+'\n'
            +'MAD = {:.1f} W'.format(MAD_t)+'\n'+'MAPE = {:.1f} %'.format(MAPE_t),loc=2)
    ax.add_artist(txt)
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    ax.set_title(str(year)+', t = '+str(t_ev[i])+' s')
    plt.savefig(path+'figures//R_'+str(t_ev[i])+'s_'+str(year)+'.png')
