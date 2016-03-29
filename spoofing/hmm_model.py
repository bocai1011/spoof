import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats

class HMM:
    def __init__(self,nState,TDFeaSet,featureSet,useAllFea,useDPGMM=True):
        '''
        recommended value for featureSet=['ewav_back buy/sell']
        '''
        self.TDFeaSet = TDFeaSet
        self.featureSet = featureSet
        self.useDPGMM = useDPGMM
        self.useAllFea = useAllFea
        #self.df = data
        self.nState = nState
        self.tp = None
        self.pi = None
        self.TDmodel = []
        self.RatioModel = []
    
    def DefStates(self,df):
        '''Define states, return data
        '''        
        s1 = {'side':'B','IsSpoof':False}
        s2 = {'side':'S','IsSpoof':False}
        s3 = {'side':'B','IsSpoof':True}
        s4 = {'side':'S','IsSpoof':True}
        df = df.loc[(df['exec sell']>0)|(df['exec buy']>0),:].copy()
        df['state']=0
        df.loc[(df['side']=='B')&(df['IsSpoof']==False),'state'] = 0
        df.loc[(df['side']=='S')&(df['IsSpoof']==False),'state'] = 1
        df.loc[(df['side']=='B')&(df['IsSpoof']==True),'state'] = 2
        df.loc[(df['side']=='S')&(df['IsSpoof']==True),'state'] = 3
        return df

    
    def fitGaussModel(self,data,showPara=True):
        ''' fits the emission model for each state
        '''
        dpgmm = GauModel()
        dpgmm.fit(data)
        if showPara:
            print '------------------'
            print '------mean--------'
            print dpgmm.normRV.mean
            print '-----Covariance----'
            print dpgmm.normRV.cov
            print '-------------------'
        return dpgmm
        #if self.useDPGMM==True:
        #    dpgmm = mixture.DPGMM(n_components=5)
        #else:
        #    dpgmm = mixture.GMM(n_components=2)
        
        #dpgmm.fit(data)
        #return dpgmm
        
    def fitGammaExpModel(self,data,showPara=True):
        ''' fits the emission model for each state, using gamma(2) pdf
        '''
        dpgmm = GammaExpModel()
        dpgmm.fit(data)
        if showPara:
            print '------------------'
            print '------sign--------'
            print dpgmm.sign
            print '-----mu----'
            print dpgmm.mu
            print '-----alpha-------'
            print dpgmm.alpha
            print '-----beta--------'
            print dpgmm.beta
            print '-------------------'
        return dpgmm
    
    def fitGammaDiracModel(self,data,showPara=True):
        ''' fits the emission model for each state, using gamma(2) pdf
        '''
        dpgmm = GammaDiracModel()
        dpgmm.fit(data)
        if showPara:
            print '------------------'
            print '------sign--------'
            print dpgmm.sign
            print '---- pi-----'
            print dpgmm.pi
            print '-----alpha-------'
            print dpgmm.alpha
            print '-----beta--------'
            print dpgmm.beta
            print '-------------------'
        return dpgmm
    
    def fitGammaModel(self,data,showPara=True):
        ''' fits the emission model for each state, using gamma(2) pdf
        '''
        dpgmm = GammaModel()
        dpgmm.fit(data)
        if showPara:
            print '------------------'
            print '------sign--------'
            print dpgmm.sign
            print '-----location----'
            print dpgmm.loc
            print '-----alpha-------'
            print dpgmm.alpha
            print '-----beta--------'
            print dpgmm.beta
            print '-------------------'
        return dpgmm
    
    def fitMixModel(self,data,showPara=True):
        ''' fits the emission model for each state
        '''
        #if self.useDPGMM==True:
        #    dpgmm = mixture.DPGMM(n_components=5)
        #else:
        dpgmm = mixture.GMM(n_components=2)        
        dpgmm.fit(data)
        
        if showPara:
            print '-------------------'
            print '------The mean------'
            print dpgmm.means_
            print '------The co-variance ---'
            print dpgmm.covars_
            print '------------------'
        return dpgmm
    
    
    def train(self,df,show=False):
        self.pi = np.array(df.groupby('state').size()*1.0/len(df))
        
        df['next state'] = df['state'].shift(-1)    
        xx = pd.DataFrame()
        for dd in df['date'].unique():
            tmp = df.loc[df['date']==dd,:].copy()
            tmp = tmp.reset_index(drop=True)
            tmp = tmp.ix[0:len(tmp)-2]
            if len(tmp)<1:
                continue
            xx = xx.append(tmp)

        gp = xx.groupby(['state','next state','date']).size()
        aa = gp.sum(level=[0,1])
        bb = gp.sum(level=0)*1.
        self.tp = aa/bb

        print '---- Transition prob'
        print self.tp
        self.TDmodel = []
        self.RatioModel = []
        
        ### RF model for ratio #############
        
        ratio0 = df.loc[df['state']==0,self.featureSet]
        ratio2 = df.loc[df['state']==2,self.featureSet]
        rf_buy = RFModel(n_estimators=10,max_depth=2)
        rf_buy.fit(ratio0,0,ratio2,2)
        
        ratio1 = df.loc[df['state']==1,self.featureSet]
        ratio3 = df.loc[df['state']==3,self.featureSet]
        rf_sell = RFModel(n_estimators=10,max_depth=2)
        rf_sell.fit(ratio1,1,ratio3,3)
        
        self.RatioModel=[RFWrapper(rf_buy,0),RFWrapper(rf_sell,1),RFWrapper(rf_buy,2),RFWrapper(rf_sell,3)]
        
        #print ratio0
        #print '------------------'
        #print ratio2
        
        #ttt = np.arange(-10,10,0.5)
        #xxx = ttt.reshape((len(ttt),1))
        #print rf_buy.prob(xxx,0)
        #print rf_buy.prob(xxx,2)
        
        #print self.RatioModel[0].prob(xxx)
        #print self.RatioModel[2].prob(xxx)
        
        #import pdb;pdb.set_trace()               
        
        for state in range(self.nState):
            #import pdb;pdb.set_trace()
            td = df.loc[df['state']==state,self.TDFeaSet]
            #ratio = df.loc[df['state']==state,self.featureSet]
            m1 = self.fitMixModel(td,showPara=False)
            self.TDmodel.append(m1)
        if show:
            self.plotDist2x(self.RatioModel[0],np.arange(-10,10,0.25),self.RatioModel[2],np.arange(-10,1e-5,0.25))
            self.plotDist2x(self.RatioModel[1],np.arange(-10,10,0.25),self.RatioModel[3],np.arange(1e-5,10,0.25))
    
    def plotDist(self,model):
        xx = np.arange(-10,10,0.25)
        yy = model.score(xx)
        plt.plot(xx,yy)
    
    def plotDist2(self,model1,model2):
        xx = np.arange(-10,10,0.25)
        yy1 = model1.score(xx)
        yy2 = model2.score(xx)
        plt.plot(xx,yy1,'*',xx,yy2,'x')
        plt.show()
    
    def plotDist2x(self,model1,x1,model2,x2):
        #xx = np.arange(-10,10,0.25)
        yy1 = model1.score(x1)
        yy2 = model2.score(x2)
        plt.plot(x1,yy1,'*',x2,yy2,'x')
        plt.show()
    
    def stateEstimator(self,obs):
        ''' Estimate the most likely state sequence given the sequence of observations
        The data (obs) should be only one day data
        '''
        nState = self.nState
        TDmodel = self.TDmodel
        RatioModel = self.RatioModel
        tdlist = []
        rtlist = []
    
        for td in TDmodel:
            tdlist.append(td.score(obs[self.TDFeaSet]))
        for rt in RatioModel:    
            rtlist.append(rt.score(obs[self.featureSet]))
    
        tdprob = np.asmatrix(tdlist)
        rtprob = np.asmatrix(rtlist)
        if self.useAllFea == True:
            distrprob = tdprob + rtprob
        else:
            distrprob = rtprob
        
        logtp = np.log(tp)
        logpi = np.log(pi)
    
        backtrack = np.ones((nState,len(obs)))*(-1)
        pathscore = np.zeros((nState,len(obs)))
        
        isbuy = obs['side'].map(lambda x:int(x=='B'))
        issell = obs['side'].map(lambda x:int(x=='S'))
        validState = np.asmatrix([isbuy,issell,isbuy,issell]) # 0 means not valid
        dumbval = -1e30
    
        ttt = np.squeeze(np.asarray(distrprob[:,0])) + logpi
        pathscore[:,0] = ttt
        for ii in range(nState):
            if validState[ii,0]==0:
                pathscore[ii,0] = dumbval
    
        for ii in range(1,len(obs)):
            for jj in range(nState):
                tmp = logtp[:,jj] + pathscore[:,ii-1]+np.squeeze(np.asarray(distrprob[:,ii]))
                pathscore[jj,ii] = max(tmp)
                backtrack[jj,ii] = np.argmax(tmp)
            for kk in range(nState):
                if validState[kk,ii]==0:
                    pathscore[kk,ii] = dumbval
                    backtrack[kk,ii] = -1
        stateSeq = [-1]*len(obs)
        stateSeq[len(obs)-1] = np.argmax(pathscore[:,len(obs)-1])
        for nn in range(len(obs)-2,-1,-1):
            stateSeq[nn] = backtrack[stateSeq[nn+1],nn+1]
        return stateSeq
    
    
    def testModelPrint(self,x):
        print 'RatioModel[0].score({}) = {}'.format(x,self.RatioModel[0].score([x]))
        print 'RatioModel[1].score({}) = {}'.format(x,self.RatioModel[1].score([x]))
        print 'RatioModel[2].score({}) = {}'.format(x,self.RatioModel[2].score([x]))
        print 'RatioModel[3].score({}) = {}'.format(x,self.RatioModel[3].score([x]))
    
    def stateProb(self,obs):
        '''Give the estimate of the probablity of each state at each time instance
        '''
        tdlist = []
        rtlist = []
    
        nState = self.nState
        TDmodel = self.TDmodel
        RatioModel = self.RatioModel
        
        if False:
            print '----Debug Info------'
            self.testModelPrint(200)
            self.testModelPrint(-200)    

        
        for td in TDmodel:
            tdlist.append(td.score(np.array(obs[self.TDFeaSet])))
        for rt in RatioModel:    
            rtlist.append(list(rt.score(np.array(obs[self.featureSet])))) #low efficiency code
        
        #import pdb;pdb.set_trace()
        tdprob = np.asmatrix(tdlist)
        rtprob = np.asmatrix(rtlist)
        if self.useAllFea == True:
            distrprob = tdprob + rtprob
        else:            
            distrprob = rtprob        
        logtp = np.log(self.tp)
        logpi = np.log(self.pi)
    
        alpha = np.zeros((nState,len(obs)))
        beta = np.zeros((nState,len(obs)))
    
        isbuy = obs['side'].map(lambda x:int(x=='B'))
        issell = obs['side'].map(lambda x:int(x=='S'))
        validState = np.asmatrix([isbuy,issell,isbuy,issell]) # 0 means not valid
        dumb = -1e5 #used to fill for np.log(zero)
    
        alpha[:,0] = np.squeeze(np.asarray(distrprob[:,0])) + logpi
        for ii in range(1,len(obs)):
            for kk in range(nState):
                if validState[kk,ii]==0:
                    alpha[kk,ii] = dumb
                else:
                    tmp = alpha[:,ii-1] + logtp[:,kk]
                    maxtmp = np.max(tmp)
                    tmp = tmp - maxtmp
                    alpha[kk,ii] = maxtmp + np.log(np.sum(np.exp(tmp))) + distrprob[kk,ii]
        
        for ii in range(len(obs)-2,-1,-1):
            for kk in range(nState):
                if validState[kk,ii] == 0:
                    beta[kk,ii] = dumb
                else:
                    tmp = np.asarray(logtp[kk])+beta[:,ii+1]+np.squeeze(np.asarray(distrprob[:,ii+1]))
                    maxtmp = np.max(tmp)
                    tmp = tmp - maxtmp
                    beta[kk,ii] = maxtmp + np.log(np.sum(np.exp(tmp)))
            
        gamma = alpha+beta # not exactly the gamma
        maxgamma = np.max(gamma,0)
        gamma = gamma - np.kron(np.reshape(maxgamma,(1,len(obs))),np.ones((nState,1)))
        gamma = np.exp(gamma)
        sumgamma = np.kron(np.sum(gamma,0),np.ones((nState,1)))
        gamma = gamma/sumgamma   
        return gamma
    
    def predict(self,df):
        ''' needs more work,better return a dataframe
        '''
        #import pdb;pdb.set_trace()
        res = pd.DataFrame()
        for xx in df['date'].unique():
            data = df.loc[df['date']==xx,:].copy()
            prob = self.stateProb(data)
            pred = np.argmax(prob,0)
            pred_prob=np.max(prob,0)
            data['pred'] = pred
            data['pred_prob'] = pred_prob
            data['predSpoofing'] = data['pred'].map(lambda x:x>1)
            res = res.append(data)
        return res
    
    def test(self,df):
        #import pdb;pdb.set_trace()
        all_truth = []
        all_score = []

        all_pred = []
        all_state = []
        
        res = pd.DataFrame()

        for xx in df['date'].unique():
            data = df.loc[df['date']==xx,:].copy()
            #import pdb;pdb.set_trace()
            prob = self.stateProb(data)
            pred = np.argmax(prob,0)
            pred_prob=np.max(prob,0)
            data['pred']=pred
            data['pred_prob']=pred_prob
            
            tmp = (np.array(pred) == np.array(data['state']))
            r = sum(tmp)*1.0/len(tmp)
            truth = map(lambda x:int(x>1),np.array(data['state']))
            score = [y if x>1 else 1-y for x,y in zip(pred,pred_prob)]
            #import pdb;pdb.set_trace()
            auc = metrics.roc_auc_score(truth,score)
            all_truth = all_truth + truth
            all_score = all_score + score
            all_pred = all_pred + list(pred)
            all_state = all_state + list(data['state'])
            res = res.append(data)
        #import pdb;pdb.set_trace()
        auc = metrics.roc_auc_score(all_truth,all_score)
        tmp = (np.array(all_pred) == np.array(all_state))
        rate = sum(tmp)*1./len(tmp)
        return auc,rate,res
    