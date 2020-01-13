import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
m = 50
n = 200
X = np.random.randn(2, m)/10

for idx in range(n):
    t = np.random.randn(2, 2)
    tmp = t[:, 0:1]/np.linalg.norm(t[:, 0]) + t[:, 1]/np.linalg.norm(t[:, 1:2])/10
    X = np.hstack((X, tmp))
    
plt.figure()
plt.scatter(X[0, m:], X[1, m:])
plt.scatter(X[0, 1:m], X[1, 1:m], c='r')

plt.show()


n = 450

H = np.eye(n) - np.ones((n,n))/n           # create centering matrix

def custom_sdist(X):
    """
    Funktion that given a matrix X returns the squared pairwise distances 
    of the column vectors in matrix form
    """
    XX = np.dot(X.T, X)
    pdists = np.outer(np.diag(XX), np.ones(XX.shape[1]).T) + np.outer(np.ones(XX.shape[0]), np.diag(XX).T) - 2*XX
    return pdists

sigma = 0.1

def K(X):
    # MISSING: kernel function for Gaussian kernel
    denominator = 2*sigma**2
    K = np.exp(-(custom_sdist(X)/denominator))	
    return K

k = 1 # number of eigenvectors

K_schlange = np.dot(H, np.dot(K(X), H))
l, V = np.linalg.eigh(K_schlange)
Sigma = np.diag(l)

Y = np.dot(Sigma[-k:,-k:], V[:, -k:].T)

fig, axs = plt.subplots(1,2,figsize=(15,6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.1)
axs = axs.ravel()
for ip in range(k):
    im = axs[ip].scatter(X[0,:], X[1,:], c=Y[ip,:])
    axs[ip].set_title('Color indicates value of PC {} at this point'.format(ip+1))
    
fig.colorbar(im)
plt.show()

plt.scatter( Y[0, :], np.zeros(Y.shape[1]))
