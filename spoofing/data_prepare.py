import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats

def toMS(x):
    return ((x.hour*60+x.minute)*60+x.second)*1000000+x.microsecond
def timeDelta2MS(x):
    return (((x.hours*60+x.minutes)*60+x.seconds)*1000+x.milliseconds)*1000+x.microseconds
    
class DataPrep:
    ''' The parameters:
    1. isLean: whether we are dealing with data with more information. isLean=False, we have a richer data like the Training set. isLean=True, we have a lean data --only brief summary for each order
    2. decay_factor: a multiple of T_M (median trading gap)
    3. linger_factor: this is the multiple of the median trading gap. With this parameter, we will ignore all ordrers placed linger_factor*T_M ago
    '''
    
    def __init__(self,isLean=False,linger_factor = 40,decay_factor=5):        
        self.isLean = isLean
        self.linger_factor = linger_factor
        self.decay_factor = decay_factor
        self.medianT = 0 #the median of the trade interval
        
    def processDatafile(self,filename):
        data = pd.read_csv(filename)
        return self.processData(data)
    
    def processData(self,data,verbose=1):
        #import pdb;pdb.set_trace()
        if verbose>0:
            print '----Data cleaning----'
        if self.isLean:            
            allorder = self.cleanDataLean(data)
        else:
            allorder = self.cleanData(data)
    
        data = self.prepare(allorder)
        
        if verbose>0:
            print '---- Feature calculation----'
        #data = self.computeEWAVForward(data)
        allorder = self.computeEWAVBackward(data)
        #data = self.computeSEV(data)
        
        if verbose>0:
            print '----- Prepare for HMM------'
        data = self.HMMPrep(allorder.copy())
        
        return allorder,data
    
    def computeEWAVBackward(self,data):
        
        if len(data)<2:
            raise ValueError('data too short')
        
        #import pdb;pdb.set_trace()
        data['time diff'] = data['time diff'].fillna(24*3600*1000000)
        self.medianT = np.median(data['time diff'])
        T = self.medianT*self.decay_factor
        linger = self.medianT*self.linger_factor
        epsilon = sys.float_info.epsilon
        
        data['ewav_back canc buy'] = epsilon
        data['ewav_back canc sell'] = epsilon
        data['ewav_back exec buy'] = epsilon
        data['ewav_back exec sell'] = epsilon
    
        for ii in range(1,len(data)):
            coef = math.exp(-data.ix[ii]['time diff']/T) if data.ix[ii]['time diff']<=linger else 0
            data.loc[ii,'ewav_back canc buy'] = data.loc[ii,'cancelled buy']+data.loc[ii-1,'ewav_back canc buy']*coef
            data.loc[ii,'ewav_back canc sell'] = data.loc[ii,'cancelled sell']+data.loc[ii-1,'ewav_back canc sell']*coef
            data.loc[ii,'ewav_back exec buy'] = data.loc[ii,'exec buy']+data.loc[ii-1,'ewav_back exec buy']*coef
            data.loc[ii,'ewav_back exec sell'] = data.loc[ii,'exec sell']+data.loc[ii-1,'ewav_back exec sell']*coef
        ff = lambda x:x if x>epsilon else epsilon
        data['ewav_back canc buy'] = data['ewav_back canc buy'].map(ff)
        data['ewav_back canc sell'] = data['ewav_back canc sell'].map(ff)
        data['ewav_back exec buy'] = data['ewav_back exec buy'].map(ff)
        data['ewav_back exec sell'] = data['ewav_back exec sell'].map(ff)
      
        data['ewav_back buy/sell'] = data['ewav_back canc buy']/data['ewav_back canc sell'] 
        data['log ewav_back buy/sell'] = data['ewav_back buy/sell'].map(math.log)
        data['ewav_back sell/buy'] = data['ewav_back canc sell']/data['ewav_back canc buy']
                
        data['ewav_back buy exec+canc'] = data['ewav_back exec buy'] + data['ewav_back canc buy']
        data['ewav_back buy exec/total']=  data['ewav_back exec buy']/data['ewav_back buy exec+canc']       
       
        data['ewav_back sell exec+canc'] = data['ewav_back exec sell'] + data['ewav_back canc sell']
        data['ewav_back sell exec/total'] = data['ewav_back exec sell']/data['ewav_back sell exec+canc']
    
        return data
    
    def computeEWAVForward(self,data):
        if len(data)<2:
            raise ValueError('data too short')
        T=np.median(data['time diff'])*2
        data['ewav_for canc buy']=0
        data['ewav_for canc sell']=0
        data['ewav_for exec buy']=0
        data['ewav_for exec sell']=0
    
        for ii in range(len(data)-2,-1,-1):
            data.loc[ii,'ewav_for canc buy'] = data.loc[ii,'cancelled buy']+data.loc[ii+1,'ewav_for canc buy']*math.exp(-data.ix[ii+1]['time diff']/T)
            data.loc[ii,'ewav_for canc sell'] = data.loc[ii,'cancelled sell']+data.loc[ii+1,'ewav_for canc sell']*math.exp(-data.ix[ii+1]['time diff']/T)
            data.loc[ii,'ewav_for exec buy'] = data.loc[ii,'exec buy']+data.loc[ii+1,'ewav_for exec buy']*math.exp(-data.ix[ii+1]['time diff']/T)
            data.loc[ii,'ewav_for exec sell'] = data.loc[ii,'exec sell']+data.loc[ii+1,'ewav_for exec sell']*math.exp(-data.ix[ii+1]['time diff']/T)
        data['ewav_for buy/sell'] = 0
        data.loc[data['ewav_for canc sell']==0,'ewav_for buy/sell']= np.inf
        data.loc[data['ewav_for canc sell']!=0,'ewav_for buy/sell'] = data.loc[data['ewav_for canc sell']!=0,'ewav_for canc buy']/data.loc[data['ewav_for canc sell']!=0,'ewav_for canc sell']
    
        data['ewav_for sell/buy'] = 0
        data.loc[data['ewav_for canc buy']==0,'ewav_for sell/buy'] = np.inf
        data.loc[data['ewav_for canc buy']!=0,'ewav_for sell/buy'] = data.loc[data['ewav_for canc buy']!=0,'ewav_for canc sell']/data.loc[data['ewav_for canc buy']!=0,'ewav_for canc buy']
    
        data['ewav_for buy exec/total'] = 0
        data['ewav_for buy exec+canc'] = data['ewav_for exec buy'] + data['ewav_for canc buy']
        data.loc[data['ewav_for buy exec+canc']==0,'ewav_for buy exec/total']= 1
        data.loc[data['ewav_for buy exec+canc']!=0,'ewav_for buy exec/total'] = data.loc[data['ewav_for buy exec+canc']!=0,'ewav_for exec buy']/data.loc[data['ewav_for buy exec+canc']!=0,'ewav_for buy exec+canc']
    
        data['ewav_for sell exec/total'] = 0
        data['ewav_for sell exec+canc'] = data['ewav_for exec sell'] + data['ewav_for canc sell']
        data.loc[data['ewav_for sell exec+canc']==0,'ewav_for sell exec/total'] = 1
        data.loc[data['ewav_for sell exec+canc']!=0,'ewav_for sell exec/total'] = data.loc[data['ewav_for sell exec+canc']!=0,'ewav_for exec sell']/data.loc[data['ewav_for sell exec+canc']!=0,'ewav_for sell exec+canc']
    
        return data
    
    def computeSEV(self,data):
        data['sev buy']=0
        data['sev sell']=0
    
        winsize = 4*np.median(data['time diff'])
        top=len(data)-1
        tail = top
        while(top>=0):
            while tail>=top:
                if data.loc[tail,'microsecond']-data.loc[top,'microsecond'] <=winsize:
                    break
                tail-=1
            tmpbuy = 0.
            tmpsell = 0.
            for ii in range(top,tail+1):
                tmpbuy += data.loc[ii,'exec buy']
                tmpsell += data.loc[ii,'exec sell']
            data.loc[top,'sev buy']=tmpbuy
            data.loc[top,'sev sell']=tmpsell
            top -= 1
        data['sev net buy'] = data['sev buy'] - data['sev sell']
        data.loc[data['sev buy']==0,'sev sell/buy'] = np.inf
        data.loc[data['sev buy']!=0,'sev sell/buy'] = data.loc[data['sev buy']!=0,'sev sell']/data.loc[data['sev buy']!=0,'sev buy']*1.
        data.loc[data['sev sell']==0,'sev buy/sell'] = np.inf
        data.loc[data['sev sell']!=0,'sev buy/sell'] = data.loc[data['sev sell']!=0,'sev buy']/data.loc[data['sev sell']!=0,'sev sell']*1.
        return data  
    
    def cleanDataLean(self,data):
        #data['q_exec'].fillna(0,inplace=True)
        data['q_exec'].fillna(0,inplace=True)
        data['execution_time'] = data['execution_time'].map(lambda x:pd.to_datetime(x))
        data['cancel_entry_time'] = data['cancel_entry_time'].map(lambda x:pd.to_datetime(x))
        data['order_entry_time'] = data['order_entry_time'].map(lambda x:pd.to_datetime(x))
        
        allorder = data
        allorder['prc*qty'] = allorder['avg_prc']        
        allorder['execution_time_last_ms'] = allorder['execution_time'].map(toMS)
        allorder['order_entry_time_ms'] = allorder['order_entry_time'].map(toMS)
        
        allorder['q_cancel'] = allorder['q_new'] - allorder['q_exec']
        allorder.set_index('orderid',inplace=True)
        allorder = allorder.sort('order_entry_time')
        return allorder
    
    def cleanData(self,data):
        data['q_exec'].fillna(0,inplace=True)
        data['execution_time'] = data['execution_time'].map(lambda x:pd.to_datetime(x))
        data['cancel_entry_time'] = data['cancel_entry_time'].map(lambda x:pd.to_datetime(x))
        data['order_entry_time'] = data['order_entry_time'].map(lambda x:pd.to_datetime(x))
        data['prc*qty'] = data['q_exec']*data['prc_exec']

        neworder = data.loc[data['order_type']=='NEW ORDER',:]
        exeorder = data.loc[data['order_type']=='EXECUTION',:]
        canorder = data.loc[data['order_type']=='CANCEL',:].copy()
    
        ############## Exclude those partial filled orders from cancel list
        #partialfill = set(canorder['orderid']).intersection(set(exeorder['orderid']))
        #canorder = canorder.loc[canorder['orderid'].isin(partialfill)==False,:]
        #####################################################################
   
        allorder = neworder[['id','orderid','symbol','q_new','price','order_entry_time','date','time','side']].set_index('orderid')
        gp = exeorder.groupby('orderid')
        tmp = gp.agg({'q_exec':np.sum,'prc*qty':np.sum})
        tmp['avg exe_prc'] = tmp['prc*qty']/tmp['q_exec']
        del tmp['prc*qty']
        allorder = allorder.join(tmp)
    #allorder = allorder.join(gp['execute_time'].agg({'first_exe_time':np.min,'last_exe_time':np.max}))
        allorder = allorder.join(gp['execution_time'].agg({'first_execution_time':np.min,'last_execution_time':np.max}))
        allorder['execution_time_first_ms'] = allorder['first_execution_time'].map(toMS)
        allorder['execution_time_last_ms'] = allorder['last_execution_time'].map(toMS)
        allorder['order_entry_time_ms'] = allorder['order_entry_time'].map(toMS)
    #gp = canorder.groupby('orderid')
        allorder = allorder.join(canorder.set_index('orderid')[['cancel_entry_time','canc_time']])
        allorder['q_exec'].fillna(0,inplace=True)
        allorder['q_cancel'] = allorder['q_new'] - allorder['q_exec']
        allorder = allorder.sort('order_entry_time')
        return allorder
    
    def prepare(self,allorder):
        ''' resort all the order according the the order entry time (canceled order) and exe time(filled order)
            Calculate the time difference
        '''
        fillorder = allorder.loc[allorder['q_exec']>0,['date','price','side','last_execution_time','execution_time_last_ms','q_exec']]
        fillorder['exec buy'] = fillorder['q_exec']
        fillorder['exec sell'] = fillorder['q_exec']
        fillorder.loc[fillorder['side']=='B','exec sell'] = 0
        fillorder.loc[fillorder['side']!='B','exec buy'] = 0
        fillorder = fillorder.rename(columns={'execution_time_last_ms':'microsecond','last_execution_time':'time'})

    #canorder = allorder.loc[allorder['q_cancel']>0,['date','price','side','order_entry_time','order_entry_time_ms','q_cancel']]
        canorder = allorder.loc[allorder['q_cancel']==allorder['q_new'],['date','price','side','order_entry_time','order_entry_time_ms','q_cancel']]
    #partially filled order discarded
    #import pdb;pdb.set_trace()
        canorder['cancelled buy'] = canorder['q_cancel']
        canorder['cancelled sell'] = canorder['q_cancel']
        canorder.loc[canorder['side']=='B','cancelled sell'] = 0.0
        canorder.loc[canorder['side']!='B','cancelled buy'] = 0.0
        canorder = canorder.rename(columns={'order_entry_time_ms':'microsecond','order_entry_time':'time'})

        fillorder['cancelled buy'] = 0
        fillorder['cancelled sell'] = 0
        canorder['exec buy'] = 0
        canorder['exec sell'] = 0
        del canorder['q_cancel']
        del fillorder['q_exec']
        data = fillorder.append(canorder)
        data = data.sort(['date','microsecond'])
        data = data.reset_index()
        #import pdb;pdb.set_trace()
        for dd in data['date'].unique():
            data.loc[data['date']==dd,'inventory'] = data.loc[data['date']==dd,'exec buy']-data.loc[data['date']==dd,'exec sell']
            data.loc[data['date']==dd,'inventory'] = data.loc[data['date']==dd,'inventory'].cumsum()
            data.loc[data['date']==dd,'time diff']= data.loc[data['date']==dd,'microsecond'].diff()*1. 
        data['time diff'] = data['time diff'].fillna(24*3600*1000000)
        return data
    
    def HMMPrep(self,df):
        #import pdb;pdb.set_trace()
        col = ['orderid','cancelled buy','exec sell','cancelled sell','exec buy','microsecond','price','side','time','date','inventory','time diff',
         'ewav_back canc buy','ewav_back canc sell','ewav_back exec buy','ewav_back exec sell','ewav_back buy/sell','ewav_back sell/buy']
        if 'IsSpoof' in df.columns:
            col +=['IsSpoof']
        df = df[col]
        del df['ewav_back exec buy']
        del df['ewav_back exec sell']
        # clean the data for ewav_back canc buy/sell and sell/buy
        # buy/sell will be just inverse of sell/buy, so we use one column buy/sell
        df.loc[(df['ewav_back canc buy']<1e-5)&(df['ewav_back canc sell']<1e-5),'ewav_back buy/sell']=1
        medianbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].median()
        maxbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].max()
        df.loc[df['ewav_back buy/sell']==np.inf,'ewav_back buy/sell'] = maxbs
        df.loc[df['ewav_back buy/sell']==0,'ewav_back buy/sell'] = 1/maxbs
        df.loc[:,'ewav_back buy/sell'] = df.loc[:,'ewav_back buy/sell'].map(np.log)
        
        ## Get the time difference, seems not contributing for now
        df['TimeDiff_back'] = np.nan
        df['TimeDiff_frwd'] = np.nan
        df['TimeDiff_min'] = np.nan
        #import pdb;pdb.set_trace()
        
        df = df.loc[(df['exec sell']>0)|(df['exec buy']>0),:].copy()
        if len(df)==0:
            return df
        buy = df.loc[df['side']=='B',:].copy()
        if len(buy)>0:
            for dd in buy['date'].unique():
            #import pdb;pdb.set_trace()
                tmp = buy.loc[buy['date']==dd,:]
                buy.loc[buy['date']==dd,'TimeDiff_back'] = buy.loc[buy['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
                buy.loc[buy['date']==dd,'TimeDiff_frwd'] = buy.loc[buy['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))
            #import pdb;pdb.set_trace()    
            buy['TimeDiff_frwd'].fillna(buy['TimeDiff_frwd'].max(),inplace=True)    
            buy['TimeDiff_back'].fillna(buy['TimeDiff_back'].max(),inplace=True)
            buy['TimeDiff_min'] = buy.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

        sell = df.loc[df['side']=='S',:].copy()
        if len(sell)>0:
            for dd in sell['date'].unique():
                tmp = sell.loc[sell['date']==dd,:]
                sell.loc[sell['date']==dd,'TimeDiff_back'] = sell.loc[sell['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
                sell.loc[sell['date']==dd,'TimeDiff_frwd'] = sell.loc[sell['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))
    
            sell['TimeDiff_frwd'].fillna(sell['TimeDiff_frwd'].max(),inplace=True)
            sell['TimeDiff_back'].fillna(sell['TimeDiff_back'].max(),inplace=True)
            sell['TimeDiff_min'] = sell.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

        newdf = buy.append(sell)
        newdf['date'] = newdf['date'].map(lambda x:pd.to_datetime(x))
        #newdf = newdf.sort(['date','microsecond'])
        df = newdf.sort()
        
        return df


    