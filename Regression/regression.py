# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

def standRegres(xArr, yArr):
    '''standReges'''
    xTx = xArr.T.dot(xArr)
    if np.linalg.det(xTx) ==0.0:
        print 'xTx is singular, cannot do inverse'
        return
    return np.linalg.inv(xTx).dot(xArr.T.dot(yArr))

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)

    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in xrange(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) ==0.0:
        print 'xTx is singular, cannot do inverse'
        return
    w = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * w


def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in xrange(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def main1():
    
    dataMat = np.loadtxt('Regression/ex0.txt', delimiter='\t')
    xArr = dataMat[:,0:2]
    yArr = dataMat[:,2]
    w =  standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr[:,1],yArr)
    y = xArr.dot(w)
    ax.plot(xArr[:,1], y)
    plt.show()

def main():
    dataMat = np.loadtxt('Regression/ex0.txt', delimiter='\t')
    xArr = dataMat[:,0:2]
    yArr = dataMat[:,2]
    yHat = lwlrTest(xArr, xArr, yArr, .003)
    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xArr[:,1], yArr,s=2,c='red')
    plt.show()

def main_abalone():
    dataMat = np.loadtxt('Regression/abalone.txt', delimiter='\t')
    xArr = dataMat[:,0:8]
    yArr = dataMat[:,-1]
    yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], .1)
    yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
    yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)

    print rssError(yArr[0:99], yHat01.T)
    print rssError(yArr[0:99], yHat1.T)
    print rssError(yArr[0:99], yHat10.T)

    yHat01T = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], .1)
    yHat1T = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
    yHat10T = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)

    print rssError(yArr[100:199], yHat01T.T)
    print rssError(yArr[100:199], yHat1T.T)
    print rssError(yArr[100:199], yHat10T.T)

    # xMat = np.mat(xArr)
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # ax.scatter(xArr[:,1], yArr,s=2,c='red')
    # plt.show()

if __name__ == '__main__':
    main_abalone()