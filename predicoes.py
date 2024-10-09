# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:49:08 2024

@author: leo
"""

import time

import matplotlib.pyplot as plt


import numpy as np

import pandas as pd
import math
import scipy
import scipy.signal as sig


import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


nome= 'MUXENERGIA'
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


''
def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


#Leitura meses

ts = pd.read_csv(nome+'.csv',sep=',')


ts['MesReferencia'] = pd.to_datetime(ts['MesReferencia'], format='%d%b%Y')

ts.sort_values(by = 'MesReferencia' , inplace = True)


tempo_total=ts.iloc[:,0].to_numpy() 

tempo_total=np.arange(0,tempo_total.shape[0])

injetada_total = 10*ts.iloc[:,2].to_numpy() 


trc_total=ts.iloc[:,1].to_numpy() 

injetada_total=moving_average(injetada_total)
trc_total=moving_average(trc_total)
tempo_total=moving_average(tempo_total)


res_time=[]
res_trc=[]
res_injetada=[]

for tref in range(49,round(tempo_total[tempo_total.shape[0]-1])): 
    
   
    #tref=58
    
    injetada=injetada_total[tref-49:tref-1]
    trc=trc_total[tref-49:tref-1]
    tempo= tempo_total[tref-49:tref-1]
    
    
    
    # plt.figure(figsize=(10,6))
    # plt.plot(tempo,injetada,'blue',label='Injetada')
    # plt.plot(tempo,trc,'red',label='TRC')
    # plt.legend();
    # plt.title(nome+'')
    # plt.grid();
    
    
    
    
    injetada_trend=lowpass(injetada, 0.05, 1)
  #  plt.plot(injetada_trend)
    
    trc_trend=lowpass(trc, 0.05, 1)
   # plt.plot(trc_trend)
    
    
    # plt.figure(figsize=(10,6))
    # plt.plot(tempo,injetada,'b-',label='Injetada')
    # plt.plot(tempo,injetada_trend,'b--',label='Tendência - Injetada')
    # plt.plot(tempo,trc,'r-',label='TRC')
    # plt.plot(tempo,trc_trend,'r--',label='Tendência - TRC')
    # plt.legend();
    # plt.title(nome+'')
    # plt.grid();
    
    
    injetada_detrend=injetada-injetada_trend
    
    trc_detrend=trc-trc_trend
    
    
    
    
    last=injetada_trend.shape[0]-1
    #inc_injetada=injetada_trend[last]-injetada_trend[last-1]
    
    #inc_trc=trc_trend[last]-trc_trend[last-1]
    
    dataset_injetada = pd.DataFrame( {'Injetada': injetada_detrend})
                                     #{'Tempo': tempo,
                                     #,'TRC':trc_detrend})
    print(dataset_injetada)
    
    
    
    
    model_inj= AutoReg(dataset_injetada, lags=[12,1], seasonal=True, period=6).fit()
    out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
    
    print(out.format(model_inj.aic, model_inj.hqic, model_inj.bic))  # métricas do modelo
    
    
    pred_injetada = model_inj.predict(start= 49, end=49)#round(tempo[len(dataset_injetada)-1]+1), end=round(tempo[len(dataset_injetada)-1]+1))
    predinjetada = pred_injetada.to_numpy()
    
    #predinjetada[0] = injetada_detrend[injetada_detrend.shape[0]-1]
    
    newtime=tref #round(tempo[len(dataset_injetada)-1]+1)
    
    
    dataset_trc = pd.DataFrame( {'TRC': trc_detrend})
                                     #{'Tempo': tempo,
                                     #,'TRC':trc_detrend})
    print(dataset_trc)
    
    
    
    
    model_trc= AutoReg(dataset_trc, lags=[12,1], seasonal=True, period=6).fit()
    out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
    
    print(out.format(model_trc.aic, model_trc.hqic, model_trc.bic))  # métricas do modelo
    
    pred_trc = model_trc.predict(start= 49, end=49) #(start=round(tempo[len(dataset_injetada)-1]+1), end=round(tempo[len(dataset_injetada)-1]+1))
    
    #newtime=len(dataset_trc)+np.arange(pred_trc.shape[0])
    
    predtrc = pred_trc.to_numpy()
    #predtrc[0] = trc_detrend[trc_detrend.shape[0]-1]
    
    
    
    
    
    
    # plt.figure(figsize=(10,6))
    # plt.plot(tempo,injetada_detrend,'b-',label='Injetada - detrended')
    # plt.plot(tempo,trc_detrend,'r-',label='TRC- detrended')
    # plt.plot(newtime,predinjetada,'w--',label='')
    # plt.title(nome+' - variações em torno da tendência')
    
    # plt.legend();
    # plt.grid();
    
    
    
    
    # plt.figure(figsize=(10,6))
    # plt.plot(tempo,injetada_detrend,'b-',label='Injetada - detrended')
    
    # plt.plot(newtime,predinjetada,'b*',label='Injetada- detrended, predicted')
    
    # plt.plot(tempo,trc_detrend,'r-',label='TRC- detrended')
    
    # plt.plot(newtime,predtrc,'r*',label='TRC- detrended, predicted')
    # plt.title(nome+' - variações em torno da tendência')
    
    # plt.legend();
    # plt.grid();
    
   
    
    
    

    
    #from statsmodels.graphics.tsaplots import plot_pacf
    #import statsmodels as sm
    
    
    #acv=sm.tsa.stattools.pacf(injetada_detrend.ravel(),nlags=round(len(injetada)/2-1),method="ywmle")
    
    
    #plot_pacf(injetada_detrend, method="ywmle",lags=round(len(injetada)/2-1))
    
    res_time.append(newtime)
    res_trc.append(trc[last]+predtrc[0]) #inc_trc+
    res_injetada.append(injetada[last]+predinjetada[0]) #+inc_injetada
    
injetada=injetada_total[0:tref]
trc=trc_total[0:tref]
tempo= tempo_total[0:tref]


plt.figure(figsize=(10,6))
plt.plot(tempo,injetada,'b-',label='Injetada x10')
plt.plot(res_time,res_injetada,'b--',label='Injetada x10 - predicted') #+inc_injetada
plt.plot(tempo,trc,'r-',label='TRC')
plt.plot(res_time,res_trc,'r--',label='TRC - predicted') #+inc_trc
plt.legend();
plt.title(nome+'')
plt.grid()

