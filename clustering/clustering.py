####################################
# PCA analysis code I found online #
####################################

import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

def dim_red_pca(X, d=0, corr=False):
    r"""
    Performs principal component analysis.

    Parameters
    ----------
    X : array, (n, d)
        Original observations (n observations, d features)

    d : int
        Number of principal components (default is ``0`` => all components).

    corr : bool
        If true, the PCA is performed based on the correlation matrix.

    Notes
    -----
    Always all eigenvalues and eigenvectors are returned,
    independently of the desired number of components ``d``.

    Returns
    -------
    Xred : array, (n, m or d)
        Reduced data matrix

    e_values : array, (m)
        The eigenvalues, sorted in descending manner.

    e_vectors : array, (n, m)
        The eigenvectors, sorted corresponding to eigenvalues.

    """
    # Center to average
    X_ = X-X.mean(0)
    # Compute correlation / covarianz matrix
    if corr:
        CO = np.corrcoef(X_.T)
    else:
        CO = np.cov(X_.T)
    # Compute eigenvalues and eigenvectors
    e_values, e_vectors = sp.linalg.eigh(CO)

    # Sort the eigenvalues and the eigenvectors descending
    idx = np.argsort(e_values)[::-1]
    e_vectors = e_vectors[:, idx]
    e_values = e_values[idx]
    # Get the number of desired dimensions
    d_e_vecs = e_vectors
    if d > 0:
        d_e_vecs = e_vectors[:, :d]
    else:
        d = None
    # Map principal components to original data
    LIN = np.dot(d_e_vecs, np.dot(d_e_vecs.T, X_.T)).T
    return LIN[:, :d], e_values, e_vectors

SN = np.array([ [1.325, 1.000, 1.825, 1.750],
                [2.000, 1.250, 2.675, 1.750],
                [3.000, 3.250, 3.000, 2.750],
                [1.075, 2.000, 1.675, 1.000],
                [3.425, 2.000, 3.250, 2.750],
                [1.900, 2.000, 2.400, 2.750],
                [3.325, 2.500, 3.000, 2.000],
                [3.000, 2.750, 3.075, 2.250],
                [2.075, 1.250, 2.000, 2.250],
                [2.500, 3.250, 3.075, 2.250],
                [1.675, 2.500, 2.675, 1.250],
                [2.075, 1.750, 1.900, 1.500],
                [1.750, 2.000, 1.150, 1.250],
                [2.500, 2.250, 2.425, 2.500],
                [1.675, 2.750, 2.000, 1.250],
                [3.675, 3.000, 3.325, 2.500],
                [1.250, 1.500, 1.150, 1.000]], dtype=float)
    
clust,labels_ = kmeans2(SN,3)    # cluster with 3 random initial clusters
# PCA on orig. dataset 
# Xred will have only 2 columns, the first two princ. comps.
# evals has shape (4,) and evecs (4,4). We need all eigenvalues 
# to determine the portion of variance
Xred, evals, evecs = dim_red_pca(SN,2)   

xlab = '1. PC - ExpVar = {:.2f} %'.format(evals[0]/sum(evals)*100) # determine variance portion
ylab = '2. PC - ExpVar = {:.2f} %'.format(evals[1]/sum(evals)*100)

# plot the clusters, each set separately
plt.figure()    
ax = plt.gca()
scatterHs = []
clr = ['r', 'b', 'k']
for cluster in set(labels_):
    scatterHs.append(ax.scatter(Xred[labels_ == cluster, 0], Xred[labels_ == cluster, 1], 
                   color=clr[cluster], label='Cluster {}'.format(cluster)))
plt.legend(handles=scatterHs,loc=4)
plt.setp(ax, title='First and Second Principle Components', xlabel=xlab, ylabel=ylab)

# plot also the eigenvectors for deriving the influence of each feature
fig, ax = plt.subplots(2,1)
ax[0].bar([1, 2, 3, 4],evecs[0])
plt.setp(ax[0], title="First and Second Component's Eigenvectors ", ylabel='Weight')
ax[1].bar([1, 2, 3, 4],evecs[1])
plt.setp(ax[1], xlabel='Features', ylabel='Weight')

plt.show()