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