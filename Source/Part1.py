import numpy as np
import sklearn
import sklearn.datasets
from matplotlib import pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


miu1 = [10,10]
sigma1 = [[4,4],[4,9]]
c1 = np.random.multivariate_normal(miu1, sigma1, 1000).T

miu2 = [22,10]
sigma2 = [[4,4],[4,9]]
c2 = np.random.multivariate_normal(miu2, sigma2, 1000).T


plt.figure(1)
plt.plot(c1[0,:], c1[1,:], 'o', markersize=8, color='orange', alpha=0.5, label='class1')
plt.plot(c2[0, :], c2[1, :], 'o', markersize=8, alpha=0.5, color='green', label='class2')
plt.show()

twoClass = np.concatenate((c1, c2), axis=1)
PCA_F = mlabPCA(twoClass.T)
plt.figure(2)
plt.plot(PCA_F.Y[0:1000,0], 'o', markersize=7, color='orange', alpha=0.5, label='class1')
plt.plot(PCA_F.Y[1000:2000,0], 'o', markersize=7, color='green', alpha=0.5, label='class2')
plt.show()



skln_pca = sklearnPCA(n_components=1)
transformS= skln_pca.fit_transform(twoClass.T)

inv = skln_pca.inverse_transform(transformS)
plt.plot(inv[0:1000,0],inv[0:1000,1], 'o', markersize=7, color='orange', alpha=0.5, label='class1')
plt.plot(inv[1000:2000,0], inv[1000:2000,1], 'o', markersize=7, color='green', alpha=0.5, label='class2')

err = ((inv - twoClass.T) ** 2).mean()
print(err)
plt.show()
