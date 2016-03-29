import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats

class GammaDiracModel:
    ''' Mixture of Gamma distribution and a Delta at zero
        Eventually, we want a mixture of Exponential distribution and a Gamma
    '''
    def __init__(self,sign=1,alpha=1.5,outlierRem = True):
        '''
        The model is f(x) = pi*exp(mu) + (1-pi)*Gamma(1.5,scale)
        alpha: alpha parameter for the Gamma model
        '''
        #self.loc = 0
        self.alpha = alpha
        self.pi = 0.2
        self.beta = 0
        self.sign = sign
        self.outlierRem = outlierRem
        self.auto = True # where to automatically determine the sign of the data
        
    def RemOutlier(self,data):
        ''' remove the outlier
        '''
        th = np.percentile(tmpdata,1)
        tmpdata = tmpdata[tmpdata>th] # we consider the data lower 2 percent as outliers
        return tmpdata
        
    def fit(self,data):
        ''' using EM algorithm to find out the parameters of the components
        '''
        #import pdb;pdb.set_trace()
        tmpdata = np.asarray(data)
        if self.auto:
            datamedian = np.median(tmpdata)
            if datamedian < 0 :
                self.sign = -1
        tmpdata = tmpdata*self.sign
        tmpdata = tmpdata[tmpdata>=0]
        if self.outlierRem:
            th1 = np.percentile(tmpdata,3)
            th2 = np.percentile(tmpdata,97)
            tmpdata = tmpdata[tmpdata>th1] # we consider the data lower 2 percent as outliers
            tmpdata = tmpdata[tmpdata<th2]
        
        self.pi = len(tmpdata[tmpdata==0])*1.0/len(tmpdata)
        self.beta = np.mean(tmpdata[tmpdata>0])/self.alpha
        if True:
            plt.hist(tmpdata,bins=100,normed=True)
            tt = np.arange(0,20,0.01)
            yy = self.pdf(tt)
            plt.plot(tt,yy)
            ttt = np.arange(0,50,0.01)
            zz = stats.gamma.pdf(ttt,self.alpha,loc=0,scale=self.beta)*(1-self.pi)
            plt.plot(ttt,zz)
            plt.show()
            
    def pdf(self,x):
        y = stats.gamma.pdf(x,self.alpha,loc=0,scale=self.beta)*(1-self.pi)
        y[x==0] = self.pi
        y[x<0] = 0
        return y
    def score(self,x):
        y = self.pdf(x)
        return np.log(y)