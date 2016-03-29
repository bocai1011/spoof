import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats

class GammaModel:
    def __init__(self,sign=1):
        #self.loc = 0
        self.alpha = 1.5
        #self.beta = 0
        self.gammaRV = None
        self.sign = sign
        
    def fit(self,data):
        #import pdb;pdb.set_trace()
        tmpdata = np.asarray(data)
        #tmpdata = tmpdata[tmpdata!=0] #we consider the data==0 as outlier
        datamedian = np.median(tmpdata)
        if datamedian < 0 :
            self.sign = -1
        tmpdata = tmpdata*self.sign
        th = np.percentile(tmpdata,1)
        tmpdata = tmpdata[tmpdata>th] # we consider the data lower 2 percent as outliers
        
        
        #fit_alpha, fit_loc, fit_beta=stats.gamma.fit(tmpdata)
        fit_alpha,fit_loc,fit_beta = self.fitFixedAlpha(tmpdata)
        self.gammaRV = stats.gamma(fit_alpha,loc=0, scale=fit_beta)
        #self.alpha = fit_alpha
        self.beta = fit_beta
        self.loc = 0
        
        if False:
            plt.hist(tmpdata,bins=100,normed=True)
            tt = np.arange(0,20,0.01)
            yy = self.gammaRV.pdf(tt)
            plt.plot(tt,yy)
            plt.show()
    
    def fitFixedAlpha(self,data):
        alpha = self.alpha
        loc = 0
        beta = np.mean(data)/alpha
        
        return alpha,loc,beta
        
        
    def score(self,x):     
        xx = np.array(x)
        #xx[xx==0] = sys.float_info.min
        res = self.gammaRV.logpdf(xx*self.sign)
        res[xx==0]=-1e30
        #print xx[res==np.inf]
        return res