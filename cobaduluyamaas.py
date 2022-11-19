import numpy as np
import getcovariant
import averageface
import matplotlib.pyplot as plt
import cv2

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
    q, r = qr(A)
    e = np.matmul(np.transpose(q), A)
    e = np.matmul(e, q)
    u = q
    res = np.diagonal(e)
    init = np.diagonal(A)
    while (np.sum(np.square(res-init)) > margin):
        init = res
        q, r = qr(e)
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
def getLinearCombination(aMat, uVec, K = 98, coefNum = 98):
    linComb = np.empty([K, 0], dtype=float)
    kBest = uVec[:, :K]

    for i in range(coefNum):
        a, b, c, d = np.linalg.lstsq(kBest, aMat[:, i], rcond=None)
        addZeros = K - a.size
        a = addKZeros(a, addZeros)
        a = np.reshape(a, [K, 1])
        linComb = np.append(linComb, a, axis=1)
    
    return linComb

def getClosest(avg, test, coefMat, kValue, uVec):
    normal = test - avg
    normal = np.reshape(normal, [65536, 1])
    testCoefficient = getLinearCombination(normal, uVec, kValue, 1)[:, 0]
    absoluteDifference = np.empty([0, 1], dtype=float)
    # Ganti biar ukuran training image bisa beragam
    for i in range(50):
        temp = np.absolute(testCoefficient - coefMat[:, i])
        tempSum = np.sum(temp)
        absoluteDifference = np.append(absoluteDifference, tempSum)
    minValue = np.amin(absoluteDifference)
    index = np.where(absoluteDifference == minValue)
    return index[0][0]


if __name__ == "__main__":
    myimg = averageface.getAvgFace(".\data\\newgray")
    testImgPath = "../ALGEO02-21109/data/newgray/CR89.png"
    testImg = cv2.imread(testImgPath, 0)
    curCov, aMat = getcovariant.getCovariant(myimg, "..\Algeo02-21109\data\\newgray")
    # print(curCov)
    # contoh = np.array(
    #     [[26, 40, 41, 54],
    #     [40, 67, 62, 83],
    #     [41, 62, 95, 70],
    #     [54, 83, 70, 126]])
    # q, r = qr(curCov)
    # print("q:\n", q.round(6))
    # print("r:\n", r.round(6))
    # M = np.matmul(q, r)
    # print("Multiplication result:\n", M.round(6))
    eVal, eVec = eigenValVec(curCov)
    # valveclist = np.linalg.eig(curCov)
    # eVal = valveclist[0]
    # eVec = valveclist[1]
    eVal, eVec = sortEigenVectors(eVal, eVec)
    # arrinds = eVal.argsort()
    # eVal = eVal[arrinds[::-1]]
    # eVec = eVec[arrinds[::-1]]
    # print("Eigenvalues:")
    # print(eVal)
    # print("Eigenvectors:")
    # print(eVec)
    uVec = getAdjustedVector(aMat, eVec)
    # np.savetxt('eigenvalue.txt', eVal, fmt='%.8f')
    # np.savetxt('adjusted_eigenvector.txt', uVec, fmt='%.8f')
    # print("\nLibrary")    D
    # eigenVal = valveclist[0]
    # eigenVec = valveclist[1]
    # eigenVal, eigenVec = sortEigenVectors(eigenVal, eigenVec)
    # arrinds = eigenVal.argsort()
    # eigenVal = eigenVal[arrinds[::-1]]
    # eigenVec = eigenVec[arrinds[::-1]]
    # uVector = getAdjustedVector(aMat, eigenVec)
    # np.savetxt('eigenvalue_lib.txt', eigenVal, fmt='%.8f')
    # np.savetxt('adjusted_eigenvector_lib.txt', uVector, fmt='%.8f')
    # print(valveclist[1])
    # uVec = np.loadtxt('adjusted_eigenvector.txt', dtype=float)
    # displayEigenFaces(uVec, 15)
    coef = getLinearCombination(aMat, uVec, 15)
    closestIdx = getClosest(myimg, testImg, coef, 15, uVec)
    print(closestIdx + 1)
    # print(coef)
    # linComb = getLinearCombination(aMat, uVec, 10)
    # q, r = qr(curCov)