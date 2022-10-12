import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import block_diag

def generate_simulated_data_cases(n,type_,m,cov=None,cov2=None,pos_corr=False,seed=0,mul_fac=None):
    # n : list of number of features
    # type_:list of type of features. 0: noisy, 1: mutual, 2: distinct
    # cov: covariance matrix of the entire data. Used to generate mutual gaussians
    # m   - number of samples
    
    
    covarince = []
    X = None
    
    if mul_fac is None:
        mul_fac = 0.01
        if pos_corr:
            mul_fac = 0.001
    
    for idx,(n_cur,type_cur) in enumerate(zip(n,type_)):
        if type_cur==0:
            X_i = np.random.randn(n_cur,m)*mul_fac*0.8
            cov_i = np.eye(n_cur)
        elif type_cur in [1,11,12]:
            n_start = X.shape[0] if X is not None else 0
            if type_cur in [1,11]:
#                 print("Entered - 1 ",n_start)
                cov_cur = cov[n_start:n_start+n_cur,n_start:n_start+n_cur]
            elif type_cur in [12]:
                cov_cur = cov2[n_start:n_start+n_cur,n_start:n_start+n_cur]
#             print(cov_cur.shape,n_start,n_cur,cov.shape)
            X_i = np.random.multivariate_normal(np.zeros(n_cur),cov_cur,m).T*mul_fac
            cov_i = cov_cur
        elif type_cur==2:
            np.random.seed(seed+idx)
            lam = np.exp(-1*np.arange(n_cur))
            lam = n_cur*lam/np.sum(lam)
            V = ortho_group.rvs(n_cur)
            cov_i = V@np.diag(lam)@V.T
            if pos_corr:
                cov_i=np.abs(cov_i)
                cov_i = cov_i@cov_i.T
            X_i = np.random.multivariate_normal(np.zeros(n_cur),cov_i,m).T*mul_fac
        
        if idx==0:
            X = X_i
        else:
            X = np.concatenate((X,X_i))
        
        covarince.append(cov_i)
    
    covarince = block_diag(*covarince)
        
    return X,covarince
            
