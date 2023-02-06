import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
import numpy as np
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


from matplotlib import pyplot as plt
import numpy as np
import os
#from PyPDF2 import PdfFileMerger
#merger = PdfFileMerger()

#Fitting an exponential curve and a half-normal curve for Estimating Root Length Curve


def func_exp(t, a, tau,c):
    return a*np.exp(-t*tau,dtype = np.float128)+c

def sigmoid(x, x0,b,L,c):
    return L*(scipy.special.expit((x-x0)*b,dtype = np.float128))+c  #expit(x) = 1/(1+exp(-b*(x-x0))).

def sigmoid_initial_Infle_point(x,y):
    xi=0
    xp=0
    x0_sig_inflec_p=0.1
    ipoints=[.35,0.45,.65,.75,.9]
    i=0
    for x0 in ipoints:
        #print(x0)
        args, cov =curve_fit(sigmoid, x, y,p0=[x0,5,2,.01],maxfev=10000000,ftol=1e-08, xtol=1e-08,gtol=.0001)       
        x0_sig_inflec_c,b_sig,L,c=args
       # print(f'x0_sig_inflec_c{x0_sig_inflec_c}')
        if x0_sig_inflec_c>1 or  x0_sig_inflec_c<0:
          
            if x0_sig_inflec_c<0:
                continue
            else:
                
                if (i==len(ipoints)-1):
                    xi=xp
                    break
                else: 
                    continue
                
        if x0_sig_inflec_c > x0_sig_inflec_p:

            xi=x0
            x0_sig_inflec_p=x0_sig_inflec_c
            
        xp=x0
        i=i+1
    
     
    return xi    
def count_consecutive(arr, sign):
    sign_dic = {'positive':1, 'negative':-1, 'zero':0, 'pos':1, 'neg':-1, 0:0}
    n = sign_dic.get(sign, -2)
    if n == -2:
        return "sign must be 'positive', 'negative', or 'zero'."

    signs = np.sign(arr)  # we only care about the sign
    # pad a with False at both sides for edge cases when array starts or ends with n
    d = np.diff(np.concatenate(([False], signs == n, [False])).astype(int))

    # subtract indices when value changes from False to True from indices where value changes from True to False
    # get max of these
    count =  np.max(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))

    # calculate starting index of longest sequence
    indexes = np.nonzero(d)
    dif = np.diff(indexes)
    i = dif[0].tolist().index(count)
    idx = indexes[0][i]

    #return f'Consecutive {sign} results: {count}, Indexes: {idx} - {idx+count-1}'
    return [idx,count]
#print(count_consecutive(y, 'zero')) #Consecutive zero results: 7, Indexes: 10 - 16


