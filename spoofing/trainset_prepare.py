import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import mixture
from sklearn import metrics
import sys
from scipy import stats

def prepareTrainSet():    
    df = pd.read_csv('labeled_sina.csv')
    df = df[['orderid','IsSpoof','cancelled buy','exec sell','cancelled sell','exec buy','microsecond','price','side','time','date','inventory','time diff',
         'ewav_back canc buy','ewav_back canc sell','ewav_back exec buy','ewav_back exec sell','ewav_back buy/sell','ewav_back sell/buy']]
    del df['ewav_back exec buy']
    del df['ewav_back exec sell']
    # clean the data for ewav_back canc buy/sell and sell/buy
    # buy/sell will be just inverse of sell/buy, so we use one column buy/sell
    
    df.loc[(df['ewav_back canc buy']<1e-5)&(df['ewav_back canc sell']<1e-5),'ewav_back buy/sell']=1
    medianbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].median()
    maxbs = df.loc[(df['ewav_back buy/sell']>0)&(df['ewav_back buy/sell']<np.inf),'ewav_back buy/sell'].max()
    df.loc[df['ewav_back buy/sell']==np.inf,'ewav_back buy/sell'] = maxbs
    df.loc[df['ewav_back buy/sell']==0,'ewav_back buy/sell'] = 1/maxbs
    df['ewav_back buy/sell'] = df['ewav_back buy/sell'].map(np.log)
    # To get the time difference between current trade and its nearest trade at the same side
    df['TimeDiff_back'] = np.nan
    df['TimeDiff_frwd'] = np.nan
    df['TimeDiff_min'] = np.nan
    df = df.loc[(df['exec sell']>0)|(df['exec buy']>0),:].copy()

    buy = df.loc[df['side']=='B',:].copy()
    for dd in buy['date'].unique():
        #import pdb;pdb.set_trace()
        tmp = buy.loc[buy['date']==dd,:]
        buy.loc[buy['date']==dd,'TimeDiff_back'] = buy.loc[buy['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
        buy.loc[buy['date']==dd,'TimeDiff_frwd'] = buy.loc[buy['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))
    #import pdb;pdb.set_trace()    
    buy['TimeDiff_frwd'] = buy['TimeDiff_frwd'].fillna(buy['TimeDiff_frwd'].max())    
    buy['TimeDiff_back'] = buy['TimeDiff_back'].fillna(buy['TimeDiff_back'].max())
    buy['TimeDiff_min'] = buy.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

    sell = df.loc[df['side']=='S',:].copy()
    for dd in sell['date'].unique():
        tmp = sell.loc[sell['date']==dd,:]
        sell.loc[sell['date']==dd,'TimeDiff_back'] = sell.loc[sell['date']==dd,'microsecond'].diff(1).map(lambda x:np.abs(x))
        sell.loc[sell['date']==dd,'TimeDiff_frwd'] = sell.loc[sell['date']==dd,'microsecond'].diff(-1).map(lambda x:np.abs(x))
    
    sell['TimeDiff_frwd'] = sell['TimeDiff_frwd'].fillna(sell['TimeDiff_frwd'].max())
    sell['TimeDiff_back'] = sell['TimeDiff_back'].fillna(sell['TimeDiff_back'].max())
    sell['TimeDiff_min'] = sell.apply(lambda x:min(x['TimeDiff_back'],x['TimeDiff_frwd']),axis=1)

    newdf = buy.append(sell)
    newdf['date'] = newdf['date'].map(lambda x:pd.to_datetime(x))
    #newdf = newdf.sort(['date','microsecond'])
    df = newdf.sort()
    #Define states
    s1 = {'side':'B','IsSpoof':False}
    s2 = {'side':'S','IsSpoof':False}
    s3 = {'side':'B','IsSpoof':True}
    s4 = {'side':'S','IsSpoof':True}

    df['state']=0
    df.loc[(df['side']=='B')&(df['IsSpoof']==False),'state'] = 0
    df.loc[(df['side']=='S')&(df['IsSpoof']==False),'state'] = 1
    df.loc[(df['side']=='B')&(df['IsSpoof']==True),'state'] = 2
    df.loc[(df['side']=='S')&(df['IsSpoof']==True),'state'] = 3
    
    return df
def prepareTrainSetRec():
    '''Each time the train data are loaded, all the features are calculated as the test set
    '''
    data = pd.read_csv('labeled_sina.csv')
    #import pdb;pdb.set_trace()
    dp = DataPrep()
    allorder = dp.computeEWAVBackward(data)
    hmmdata = dp.HMMPrep(allorder.copy())
    
    hmmdata['state']=0
    hmmdata.loc[(hmmdata['side']=='B')&(hmmdata['IsSpoof']==False),'state'] = 0
    hmmdata.loc[(hmmdata['side']=='S')&(hmmdata['IsSpoof']==False),'state'] = 1
    hmmdata.loc[(hmmdata['side']=='B')&(hmmdata['IsSpoof']==True),'state'] = 2
    hmmdata.loc[(hmmdata['side']=='S')&(hmmdata['IsSpoof']==True),'state'] = 3
    return hmmdata