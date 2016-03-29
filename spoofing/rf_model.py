import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

class RFModel:
    def __init__(self,n_estimators,max_depth):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.label_map = {}
        self.rev_map = {}
        
        #self.label_set = label_set
        
    def fit(self,x1,label1,x2,label2):
        ''' we assume x1,x2 are numpy arrays (1-d)
        '''
       
        label = np.array([0]*len(x1)+[1]*len(x2))
        self.label_map = {0:label1,1:label2}
        self.rev_map ={label1:0,label2:1}
        data = np.concatenate((x1,x2)).reshape((len(x1)+len(x2),1))
        self.rf.fit(data,label)
        
        if True:
            self.showResult(x1,label1,x2,label2)
    
    def showResult(self,x1,label1,x2,label2):
        #import pdb;pdb.set_trace()
        plt.hist(np.array(x1),bins=100,alpha=0.5,normed=True)
        plt.hist(np.array(x2),bins=100,alpha=0.5,normed=True)
        tt = np.arange(-50,50,0.05)
        tt = tt.reshape((len(tt),1))
        proba = self.rf.predict_proba(tt)
        plt.plot(tt,proba[:,0],color='b')
        plt.plot(tt,proba[:,1],color='r')
        plt.show()
        
    def score(self,x,label):
        ''' give score in log prob for the class denoted by label
        '''
        proba = self.rf.predict_proba(np.array(x).reshape((len(x),1)))
        return np.log(proba[:,self.rev_map[label]])
    
    def prob(self,x,label):
        ''' give score in log prob for the class denoted by label
        '''
        proba = self.rf.predict_proba(np.array(x).reshape((len(x),1)))
        return proba[:,self.rev_map[label]]
    
    
class RFWrapper():
    def __init__(self,rf,label):
        self.rf = rf
        self.label= label
    def score(self,x):
        return self.rf.score(x,self.label)
    def prob(self,x):
        return self.rf.prob(x,self.label)
    