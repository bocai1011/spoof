import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats
from sklearn.covariance import MinCovDet
from scipy.stats import multivariate_normal

class GammaExpModel:
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
        self.mu = 0
        self.beta = 0
        self.gammaRV = None
        self.ExpRV = None
        self.pi = 0.5
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
        
        nIter = 0
        self.mu = np.percentile(tmpdata,10)
        self.beta =  np.percentile(tmpdata,50)/self.alpha #initial values
        
        while nIter<100:            
            p0 = stats.expon.pdf(tmpdata,scale=self.mu)*self.pi
            p1 = stats.gamma.pdf(tmpdata,a = self.alpha,loc=0,scale=self.beta)*(1-self.pi)
            p = p0+p1
            r = p0/p
            
            #for ii in range(len(p0)):
            #    print 'ii={}, r={}'.format(ii,p0[ii]/p[ii])
            #    if p0[ii]/p[ii] is np.nan:
            #        print p0[ii],p[ii],p1[ii]
            
            old_mu = self.mu
            old_beta = self.beta
            self.mu = np.sum(r*tmpdata )/np.sum(r)
            self.beta = np.sum((1-r)*tmpdata)/np.sum(1-r)/self.alpha
            self.pi = np.mean(r)
            #print self.mu,self.beta
            if np.abs(self.mu - old_mu)<sys.float_info.epsilon and np.abs(self.beta -old_beta)<sys.float_info.epsilon:
                break
            nIter +=1
        
        #import pdb;pdb.set_trace()
        #print nIter
        if False:
            plt.hist(tmpdata,bins=100,normed=True)
            tt = np.arange(0,20,0.01)
            yy = self.pdf(tt)
            plt.plot(tt,yy)
            ttt = np.arange(0,50,0.01)
            zz = stats.gamma.pdf(ttt,self.alpha,loc=0,scale=self.beta)*(1-self.pi)
            plt.plot(ttt,zz)
            plt.show()
            
    def pdf(self,x):
        y = stats.expon.pdf(x,scale=self.mu)*self.pi + stats.gamma.pdf(x,self.alpha,loc=0,scale=self.beta)*(1-self.pi)
        return y
    def score(self,x):
        y = self.pdf(x)
        return np.log(y)
    
#    def fitFixedAlpha(self,data):
#        alpha = self.alpha
#        loc = 0
#        beta = np.mean(data)/alpha
        
#        return alpha,loc,beta
        
        
#    def score(self,x):     
#        xx = np.array(x)
#        #xx[xx==0] = sys.float_info.min
#        res = self.gammaRV.logpdf(xx*self.sign)
#        res[xx==0]=-1e30
#        #print xx[res==np.inf]
#        return res