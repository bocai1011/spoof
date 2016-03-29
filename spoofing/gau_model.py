from sklearn.covariance import MinCovDet
from scipy.stats import multivariate_normal

class GauModel:
    def __init__(self):
        #self.mu = np.None
        #self.sigma = np.None
        self.normRV = None
        
    
    def fit(self,data):
        robust_cov = MinCovDet().fit(data)
        self.normRV = multivariate_normal(mean = robust_cov.location_,cov=robust_cov.covariance_)
    
    def score(self,x):
        return self.normRV.logpdf(x)