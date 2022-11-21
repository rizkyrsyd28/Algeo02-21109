import numpy as np
import getcovariant
import averageface
import matplotlib.pyplot as plt
import cv2
from os.path import exists
import os
import natsort
import time
import convert2gray

def qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def eigenValVec(A, margin = 1e-20):
    q, r = np.linalg.qr(A)
    e = np.matmul(np.transpose(q), A)
    e = np.matmul(e, q)
    u = q
    res = np.diagonal(e)
    init = np.diagonal(A)
    while (np.sum(np.square(res-init)) > margin):
        init = res
        q, r = np.linalg.qr(e)
        e = np.matmul(np.transpose(q), e)
        e = np.matmul(e, q)
        u = np.matmul(u, q)
        res = np.diagonal(e)
    res = np.around(np.diagonal(e), 6)
    vectors = u
    return (res, vectors)

def sortEigenVectors(eigenVal, eigenVec):
    arrIdx = eigenVal.argsort()
    eigenValRes = eigenVal[arrIdx[::-1]]
    eigenVecRes = np.copy(eigenVec)
    x, y = eigenVec.shape
    for i in range(x):
        eigenVecRes[i, :] = (eigenVec[i, :])[arrIdx[::-1]]
    return eigenValRes, eigenVecRes

def getAdjustedVector(mat, eVec):
    res = np.empty([65536, 0], dtype=float)
    x, y = eVec.shape

    for i in range(y):
        temp = np.matmul(mat, eVec[:, i])
        temp = np.reshape(temp, [65536, 1])
        res = np.append(res, temp, axis=1)
    
    return res

def displayEigenFaces(uVec, k):
    # Ganti biar ukuran training image bisa beragam
    adjusted = np.empty([256, 256, 0], dtype=float)
    for i in range(k+1):
        temp = np.reshape(uVec[:, i], [256, 256, 1])
        adjusted = np.append(adjusted, temp, axis=2)
    figs, axs = plt.subplots(3, 5)
    for i, ax in enumerate(axs.flatten()):
        if i < k:
            ax.imshow(adjusted[:, :, i])
        else:
            ax.remove()
    plt.show()

def addKZeros(arr, K):
    res = np.copy(arr)
    for i in range(K):
        res = np.append(res, 0, axis=1)
    return res

# Ganti biar ukuran training image bisa beragam
def getLinearCombination(aMat, uVec, K):
    kBest = uVec[:, :K]

    a, b, c, d = np.linalg.lstsq(kBest, aMat, rcond=None)
    
    return a

def getClosest(avg, test, coefMat, kValue, uVec):
    normal = np.subtract(test, avg)
    normal = np.reshape(normal, [65536, 1])
    testCoefficient = getLinearCombination(normal, uVec, kValue)[:, 0]
    absoluteDifference = np.empty([0, 1], dtype=float)
    x, y = coefMat.shape
    for i in range(y):
        temp = testCoefficient - coefMat[:, i]
        tempSum = np.linalg.norm(temp)
        absoluteDifference = np.append(absoluteDifference, tempSum)
    minValue = np.amin(absoluteDifference)
    index = np.where(absoluteDifference == minValue)
    return index[0][0], minValue

def predictImageIndex(testImage, trainingPath):
    avgImg, imgNum = averageface.getAvgFace(trainingPath)
    testImg = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    testImg = cv2.resize(testImg, (256, 256))
    covariant, AMatrix = getcovariant.getCovariant(avgImg, trainingPath)
    eigenValArrayPath = "../parahlimpik/eigen/eigenValue.txt"
    adjustedEigenVecArrayPath = "../parahlimpik/eigen/eigenVec.txt"
    startTime = time.time()
    if (exists(eigenValArrayPath) and exists(adjustedEigenVecArrayPath)):
        uVec = np.loadtxt(adjustedEigenVecArrayPath, dtype=float)
        eVal = np.loadtxt(eigenValArrayPath, dtype=float)
    else:
        eVal, eVec = eigenValVec(covariant)
        eVal, eVec = sortEigenVectors(eVal, eVec)
        uVec = getAdjustedVector(AMatrix, eVec)
        np.savetxt(eigenValArrayPath, eVal, fmt='%.20f')
        np.savetxt(adjustedEigenVecArrayPath, uVec, fmt='%.20f')
    endTime = time.time() - startTime
    coef = getLinearCombination(AMatrix, uVec, imgNum)
    closestIdx, minValue = getClosest(avgImg, testImg, coef, imgNum, uVec)
    resultPath = getImagePath(closestIdx, trainingPath)
    return minValue, endTime, resultPath

def getImagePath(index, path):
    fileList = []
    for filePath in os.listdir(path):
        if os.path.isfile(os.path.join(path, filePath)):
            fileList.append(filePath)

    fileList = natsort.natsorted(fileList, key=lambda y: y.lower())
    fileName = fileList[index]
    filePath = path + fileName
    return filePath


if __name__ == "__main__":
    # myimg = averageface.getAvgFace("../ALGEO02-21109/data/gray/")
    trainingDir = "../ALGEO02-21109/data/Face Recognition/"
    testImgPath = "../ALGEO02-21109/data/test/IMG_5754.jpg"
    testImg = cv2.imread(testImgPath)
    res, t, path = predictImageIndex(testImg, trainingDir)
    print(res)
    print(t)
    print(path)