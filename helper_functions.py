from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as pathces

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import scprep
import phate

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def compute_diff_vectors(X_A,X_B,k,d1,d2=None,full_matrices=True, symmetric=False):
    d2 = d1 if d2 is None else d2
    P_A,lam_A,V_A = compute_spectral_embedding(X_A,k,d1,symmetric=symmetric)
    P_B,lam_B,V_B = compute_spectral_embedding(X_B,k,d2,symmetric=symmetric)

    p = P_A.shape[0]
    Q_A = np.eye(p)-V_A@(np.linalg.inv(V_A.T@V_A)@V_A.T)
    Q_B = np.eye(p)-V_B@(np.linalg.inv(V_B.T@V_B)@V_B.T)

    U_A_diff,lam_A_diff,V_A_diff = np.linalg.svd(P_A@Q_B,full_matrices=full_matrices)
    U_B_diff,lam_B_diff,V_B_diff = np.linalg.svd(P_B@Q_A,full_matrices=full_matrices)
    U_A_diff = U_A_diff.T
    U_B_diff = U_B_diff.T
    sign = np.where(np.argmax(V_A_diff,axis=-1)==np.argmax(np.abs(V_A_diff),axis=-1),1,-1)
    V_A_diff = V_A_diff/(sign[:,None])
    
    sign = np.where(np.argmax(V_B_diff,axis=-1)==np.argmax(np.abs(V_B_diff),axis=-1),1,-1)
    V_B_diff = V_B_diff/(sign[:,None])
    
    sign = np.where(np.argmax(U_A_diff,axis=-1)==np.argmax(np.abs(U_A_diff),axis=-1),1,-1)
    U_A_diff = U_A_diff/(sign[:,None])
    
    sign = np.where(np.argmax(U_B_diff,axis=-1)==np.argmax(np.abs(U_B_diff),axis=-1),1,-1)
    U_B_diff = U_B_diff/(sign[:,None])
    
    results = {"P_A":P_A,"lam_A":lam_A,"V_A":V_A,
               "P_B":P_B,"lam_B":lam_B,"V_B":V_B,
               "Q_A":Q_A,"Q_B":Q_B,
               "U_A_diff":U_A_diff,"lam_A_diff":lam_A_diff,"V_A_diff":V_A_diff,
               "U_B_diff":U_B_diff,"lam_B_diff":lam_B_diff,"V_B_diff":V_B_diff}
    return results

def compute_diff_vectors_n(X,k,d,full_matrices=True,symmetric=False):
    n = len(X)
    if isinstance(d,int):
        d = [d for i in range(n)]
    
    P,lam,V = [],[],[]
    for i in range(n):
        P_cur,lam_cur,V_cur = compute_spectral_embedding(X[i],k,d[i],symmetric=symmetric)
        P.append(P_cur)
        lam.append(lam_cur)
        V.append(V_cur)
    
    M,Q,U_diff,lam_diff,V_diff = [],[],[],[],[]
    p = X[0].shape[-1]
    for i in range(n):
        M_cur = np.hstack([V[j] for j in range(n) if j!=i])
        Q_cur = np.eye(p)-M_cur@(np.linalg.inv(M_cur.T@M_cur)@M_cur.T)
        U_cur_diff,lam_cur_diff,V_cur_diff = np.linalg.svd(P[i]@Q_cur,full_matrices=full_matrices)
        
        sign = np.where(np.argmax(V_cur_diff,axis=-1)==np.argmax(np.abs(V_cur_diff),axis=-1),1,-1)
        V_cur_diff = V_cur_diff/(sign[:,None])
        
        sign = np.where(np.argmax(U_cur_diff,axis=-1)==np.argmax(np.abs(U_cur_diff),axis=-1),1,-1)
        U_cur_diff = U_cur_diff/(sign[:,None])
        
        M.append(M_cur)
        Q.append(Q_cur)
        U_diff.append(U_cur_diff)
        lam_diff.append(lam_cur_diff)
        V_diff.append(V_cur_diff)
    
    results = {"P":P,"lam":lam,"V":V,
               "M":M,"Q":Q,
               "U_diff":U_diff,"lam_diff":lam_diff,"V_diff":V_diff}
    return results

# pairwise distance
def compute_distances(X,Y=None):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X,Y       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''
    if Y is None:
        # return distance matrix
#         D = np.linalg.norm(X[:,:, np.newaxis] - X[:,:, np.newaxis].T, axis = 1)
        D = distance.cdist(X,X,metric='euclidean')
    else:
        D = distance.cdist(X,Y,metric='euclidean')
#         D = np.linalg.norm(X[:,:, np.newaxis] - Y[:,:, np.newaxis].T, axis = 1)
    
    
    return D



def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''

    # return the affinity matrix
    if kernel_type == "gaussian":
        if sigma is None:
            raise ValueError('sigma must be provided for gaussian kernel.')
        elif ((type(sigma) is not int) and (type(sigma) is not float)) or sigma <= 0:
            raise ValueError('sigma must be a positive number.')
            
        W = np.exp(- D ** 2 / (2 * sigma ** 2)) 
        
    elif kernel_type == "adaptive":
        if k is None:
            raise ValueError('k must be provided for adaptive gaussian kernel.')
        elif (type(k) is not int) or k <= 0:
            raise ValueError('k must be a positive integer.')
        # compute sigma 
        D_copy = D.copy()
        D_copy = D + np.diag(np.repeat(float("inf"), D.shape[0]))
        nn = D_copy.argsort(axis = 1)

        kth_idx = nn[:,k-1]
        kth_dist = D[range(D.shape[0]), kth_idx]
        s_i = np.tile(kth_dist[:,np.newaxis], (1, D.shape[0]))
        W = 1 / 2 * (np.exp(- D ** 2 / (s_i**2)) + np.exp(- D ** 2 / (s_i.transpose()**2)))

    elif kernel_type == "adaptive_sparse":
        if k is None:
            raise ValueError('k must be provided for adaptive gaussian kernel.')
        elif (type(k) is not int) or k <= 0:
            raise ValueError('k must be a positive integer.')
        # compute sigma 
        D_copy = D.copy()
        D_copy = D + np.diag(np.repeat(float("inf"), D.shape[0]))
        nn = D_copy.argsort(axis = 1)
        kth_idx = nn[:,k-1]
        W = np.zeros(D.shape)        
        for i in range(W.shape[0]):
            W[i,nn[i,0:k]] = np.exp(- D[i,nn[i,0:k]] ** 2 / (D[i,nn[i,k]]**2)) 
        W = 1/2*(W+W.transpose())
        #s_i = np.tile(kth_dist[:,np.newaxis], (1, D.shape[0]))
        #W = 1 / 2 * (np.exp(- D ** 2 / (s_i**2)) + np.exp(- D ** 2 / (s_i.transpose()**2)))


    else:
        raise ValueError('kernel_type must be either "gaussian" or "adaptive".')
        
    return W
 

    

def compute_sparse_knn_matrix(X,k):
    m = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=(k+1)).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:,1:]
    indices = indices[:,1:]
    rows = np.repeat(range(m),k)
    cols = np.matrix.flatten(indices)
    kth_idx = indices[:,k-1]
    kth_dist = distances[:,k-1]
    s_i = np.tile(kth_dist[:,np.newaxis],k)
    W = np.exp(- distances ** 2 / (s_i**2))
    W_vec = np.matrix.flatten(W)
    K = scipy.sparse.csr_matrix((W_vec, (rows, cols)), shape=(m, m))
    K = K+K.T
    return K

def compute_spectral_embedding_sparse(K,d):
    row_sum = np.sum(W, axis = 1)    
    D_inv = scipy.sparse.spdiags(1/row_sum, 0, row_sum.size, row_sum.size)
    P = D_inv@K
    lmbda,v = scipy.sparse.linalg.eigs(P,d)
    return lmbda,v


def compute_spectral_embedding(X, k, d, symmetric=False):
    
    # compute weight matrix W    
    X_dist = compute_distances(X.T)
    
    X_aff = compute_affinity_matrix(X_dist, "adaptive", k = k)    
    W = X_aff #- np.eye(X.T.shape[0])    
    

    # Compute random walk Laplacian matrix with density normalization
    if symmetric:
        Ms = W
        row_sum = np.sum(Ms, axis = 1,keepdims=True)
        row_sum = np.sqrt(row_sum)
        col_sum = np.sum(Ms, axis = 0,keepdims=True)
        col_sum = np.sqrt(col_sum)
#         D_inv = np.diag(1/row_sum)
        Ms = (Ms/row_sum)/col_sum
    else:
        row_sum = np.sum(W, axis = 1)
        D_inv = np.diag(1/row_sum)
        Ms = D_inv @ W @ D_inv
        row_sum = np.sum(Ms, axis = 1)
        D_inv = np.diag(1/row_sum)
        Ms = D_inv @ Ms

    # Compute spectral eigenvectors of the graph Laplacian
    lmbda, v = np.linalg.eig(Ms)
    order = np.argsort(lmbda)[::-1]
    lmbda = lmbda[order]
    v = v[:,order]
    lmbda = lmbda[1:(d+1)]
    v = v[:,1:(d+1)]
    sign = np.where(np.argmax(v,axis=0)==np.argmax(np.abs(v),axis=0),1,-1)
    v = v/(sign[None,:])
#     v = v/np.sign(v[0,:])
    return Ms,lmbda,v

def compute_leading_eigenvectors(P,d,sym_flag):
    if sym_flag == 1:
        lmbda, v = np.linalg.eigh(P)
        order = np.argsort(lmbda)[::-1]
        lmbda = lmbda[order]
        v = v[:,order]
        lmbda = lmbda[0:d]
        v = v[:,0:d]
        v = v/np.sign(v[0,:])
    else:
        lmbda, v = np.linalg.eig(P)
        order = np.argsort(lmbda)[::-1]
        lmbda = lmbda[order]
        v = v[:,order]
        lmbda = lmbda[0:d]
        v = v[:,0:d]
        v = v/np.sign(v[0,:])
    return lmbda,v
    
