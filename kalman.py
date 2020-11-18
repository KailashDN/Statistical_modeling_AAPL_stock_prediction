#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:14:55 2019

@author: mendon
"""

import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
df = pd.read_excel('AAPL.xlsx',sheetnames = 'AAPL')

def Kalman_Filter(Y):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = param0[0]
 T = param0[1]
 H = param0[2]
 Q = param0[3]
 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 KF_Dens = np.zeros(S)
 for s in range(1,S):
  if s == 1:
    P_update[s] = 1000
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q
  else:
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
    v[s]=Y[s-1]-Z*u_predict[s-1]
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s];
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q
    KF_Dens[s] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]
    Likelihood = np.sum(KF_Dens[1:-1])
   
    return Likelihood


def Kalman_Smoother(params, Y):
 S = Y.shape[0]
 S = S + 1
 "Initialize Params:"
 Z = params[0]
 T = params[1]
 H = params[2]
 Q = params[3]

 "Kalman Filter Starts:"
 u_predict = np.zeros(S)
 u_update = np.zeros(S)
 P_predict = np.zeros(S)
 P_update = np.zeros(S)
 v = np.zeros(S)
 F = np.zeros(S)
 for s in range(1,S):
   if s == 1:
    P_update[s] = 1000
    P_predict[s] =  T*P_update[1]*np.transpose(T)+Q
   else:
    F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
    v[s]=Y[s-1]-Z*u_predict[s-1]
    u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
    u_predict[s] = T*u_update[s];
    P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1];
    P_predict[s] = T*P_update[s]*np.transpose(T)+Q

    u_smooth = np.zeros(S)
    P_smooth = np.zeros(S)
    u_smooth[S-1] = u_update[S-1]
    P_smooth[S-1] = P_update[S-1]
    
 for  t in range(S-1,0,-1):
        u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[t])
        P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]
 u_smooth = u_smooth[0:-1]
 return u_smooth


def rmse_calc(actual, predicted):
    sum_error = 0.0
    for i in range(Y.shape[0]):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(Y.shape[0])
    return np.sqrt(mean_error)



B = df['Close ']
A = df['Open ']
C = df['High ']
D = df['Low ']

Y = B
T = Y.shape[0]

param0 = np.array([1, 0.45, 10*np.var(Y), 300*np.var(Y)])
param_star = minimize(Kalman_Filter, param0, method='BFGS',options={'gtol': 1e-8, 'disp': False})
Y_update = Kalman_Smoother(param_star.x, Y)
timevec = df[1:]['Date']
plt.plot(timevec, Y_update[1:],'r',timevec, Y[1:],'b:')
print(rmse_calc(Y,Y_update))