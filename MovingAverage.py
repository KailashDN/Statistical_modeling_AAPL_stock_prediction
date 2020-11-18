#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:24:08 2019

@author: nirmaljay
"""

import numpy as np
import scipy.stats as ss
import pandas  as pd
df = pd.read_excel('AAPL.xlsx',sheetnames = 'AAPL')

B = df['Close ']
A = df['Open ']
C = df['High ']
D = df['Low ']

Y = (A+B+C+D)/4

T = Y.shape[0]
def Moving_Avg(Y):
    rolling_mean = Y.rolling(window=50).mean()
    
    ema_short = Y.ewm(span=20, adjust=False).mean()
    """
    plt.plot(Y,label='APPL')
    
    plt.plot(trading_positions,label='APPL 50 Day SMA', color='red')
    plt.legend(loc='upper left')
    plt.show()
    """
    return ema_short

def MovingAverageLR(B):
    mu = (0,0,0)
    cov = [[1, 0, 0],\
           [0, 1, 0],\
           [0, 0, 1]]

    F = np.random.multivariate_normal(mu,cov,T)

    'Generate Sample for Y,X'
    X  = B.ewm(span=20, adjust=False).mean().as_matrix()
    'N =(750,4),N[1] = 4'
    N = X.shape
    'Define Coefficient Matrix , define rank to be N[1] x 1'
    beta = np.array([0.56,2.53,2.05,1.78])
    
    'Generate a Sample of Y'
    Y = X@beta+ np.random.normal(0,1,(T,1))

    'OLS Regression Starts'
    'Linear Regression of Y: T x 1 on'
    'Regressors X: T x N'
    invXX = np.linalg.inv(X.transpose()@X)
    'OLS estimates for coefficients: X x 1'
    beta_hat = invXX@X.transpose()@Y

    'Predictive value of Y using OLS'
    y_hat = X@beta_hat
    'Residuals from OLS'
    residuals = Y - y_hat
    'Variance of Residuals'
    sigma2 = (1/T)*residuals.transpose()@residuals
    'standard deviation of Y or residuals'
    sigma = np.sqrt(sigma2)
    'variance-covaRIANCE matrix of beta_hat'
    varcov_beta_hat = (sigma2)*invXX
    std_beta_hat = np.sqrt(T*np.diag(varcov_beta_hat))

    'Calculate R-square'
    R_square = (residuals.transpose()@residuals)/(T*np.var(Y))
    adj_R_square = 1 - (1 - R_square)*(T-1)/(T-N[1])

    'Test Each Coefficient: beta_i'
    'Null Hypothesis:beta_i = 0'
    t_stat = (beta_hat.transpose() - 0)/std_beta_hat
    p_val = 1-ss.norm.cdf(t_stat)

    'Test of Joint Significance of Modal'
    F_stat = (beta_hat.transpose()@np.linalg.inv(varcov_beta_hat)@beta_hat/N[1]/\
          (residuals.transpose()@residuals)/(T-N[1]))
    p_val_F = 1 - ss.f.cdf(F_stat,N[1]-1,T-N[1])
    


MovingAverageLR(Y)