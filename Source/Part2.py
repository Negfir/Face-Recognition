from itertools import product

from PIL import  Image
from builtins import print
from numpy import  array
import  numpy as np
import matplotlib.pylab as plt
from scipy.misc import toimage
from sklearn.linear_model import LogisticRegression

x_train = []
y_train = []
mean = np.zeros((1, 2500))

with open('train.txt') as f:
    l = f.readline()
    counter = 1
    while l:
        adrs = l.split(' ')
        im = plt.imread(adrs[0])
        final = np.reshape(im,(1,2500))
        x_train.append(final[0])
        y_train.append(adrs[1])
        mean += final
        l = f.readline()
        counter += 1
mean /= len(x_train)

x_test = []
y_test = []

with open('test.txt') as f:
    l = f.readline()
    counter = 1
    while l:
        adrs = l.split(' ')
        im = plt.imread(adrs[0])
        final = np.reshape(im, (1, 2500))
        x_test.append(final)
        y_test.append(adrs[1])
        l = f.readline()
        counter += 1


train_mat = []
t = None
for t in x_train:
    t = t - mean
    trainf = []
    for i in t[0]:
        trainf.append(i)
    train_mat.append(trainf)
toimage(np.matrix.reshape(x_train[258],50,50)).show()
toimage(np.matrix.reshape(t,50, 50)).show()



test_mat = []
for t in x_test:
    t = t - mean
    test_mat.append(t)


U, sigma, Vt = np.linalg.svd(train_mat, full_matrices=False)

for i in range(11):
    t = np.matrix.reshape(Vt[i],50,50)
    im = toimage(t).show()


x_r = []

check = len(sigma)
for r in range (1,200):
    print(r)
    sigma2 = np.zeros((check,check))
    for i in range(r):
        sigma2[i][i] = sigma[i]
    frt = np.matmul(U,sigma2)
    xr = np.matmul(frt,Vt)
    er = np.linalg.norm(xr - x_train)
    x_r.append(er)



plt.plot(x_r)
plt.ylabel('error')
plt.xlabel('r')
plt.show()


F = np.matmul(x_train, Vt[:9,:].T)
FT = np.matmul(x_test, Vt[:9,:].T)
eigen = []
for f in range(len(FT)):
    eigen.append(FT[f][0])
toimage(F).show()
toimage(eigen).show()



err = []
for r in range(1,200):
    F = np.matmul(x_train, Vt[:r,:].T)
    FT = np.matmul(x_test, Vt[:r,:].T)
    eigen = []
    for f in range(len(FT)):
        eigen.append(FT[f][0])



    Regres = LogisticRegression(C=1e5)
    Regres.fit(F, y_train)

    estimated_y = Regres.predict(eigen)

    sum = 0

    for i in range(len(y_test)):
        if int(y_test[i]) != int(estimated_y[i]):
            sum += 1
    err.append(sum)

plt.plot(err)
plt.ylabel('error')
plt.xlabel('r')
plt.show()
