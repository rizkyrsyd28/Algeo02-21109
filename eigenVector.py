import numpy as np
from qrFactorization import *
from tabulate import tabulate

def eigenVector(Cov, row, col) :

    Q = qrFactorization(Cov, row, col) 
    E = np.matmul(np.transpose(Q),Cov)
    E = np.matmul(E,Q)
    V = Q

    init = np.diag(Cov)
    res = np.diag(E)
    sum = np.sum(np.square(res-init))
    while (sum > 1e-15) :
        init = res

        Q = qrFactorization(E, row, col) 
        E = np.matmul(np.transpose(Q),Cov)
        E = np.matmul(E,Q)
        V = np.matmul(V,Q)

        res = np.diag(E)
        sum = np.sum(np.square(res-init))

    return V

def eigenVectorPakeNumpy(Cov, row, col) :

    Q, R = np.linalg.qr(Cov) 
    E = np.matmul(np.transpose(Q),Cov)
    E = np.matmul(E,Q)
    V = Q

    init = np.diag(Cov)
    res = np.diag(E)
    sum = np.sum(np.square(res-init))
    while (sum > 1e-15) :
        init = res

        Q, NULL= np.linalg.qr(E) 
        E = np.matmul(np.transpose(Q),E)
        E = np.matmul(E,Q)
        V = np.matmul(V,Q)

        res = np.diag(E)
        sum = np.sum(np.square(np.subtract(res,init)))

    return V

def eigenVectorNew(Cov) :
    CovCopy = np.copy(Cov)
    row, col = Cov.shape
    I = np.eye(row)
    Qn = I

    for i in range(10000) :
        Q = qrFactorization(Cov, row, col)
        R = getR(Q, Cov)
        Cov = np.matmul(Cov,R)
        Qn = np.matmul(Qn,Q)

    eigenVal = np.diag(Qn)
    eigenVec = np.empty(shape=(row,col), dtype = float)
    zeros = np.zeros(shape=(0,col), dtype = int)
    eigenVal = np.sort(eigenVal)[::-1]

    for i in range(eigenVal.shape) :
        lambdaI = np.matmul(eigenVal,I)
        eigenVector[:,i] = np.solve(np.substract(lambdaI,CovCopy),zeros)

    return eigenVector

if __name__ == "__main__":
    # img = avgFace()
    # covid = getCov(img)
    # V = np.empty(shape=(256, 256), dtype=int)

    # row = 256
    # col = 256

    res = np.empty(shape=(4, 4), dtype=int)
    res = inputFromKeyBoard(4,4)
    row = 4
    col = 4

    Q, R = np.linalg.qr(res)


    # V = eigenVectorPakeNumpy(res, row, col)
    VAL, V1 = np.linalg.eig(res)

    print(tabulate(Q))
    print(tabulate(R))
    print(tabulate(np.matmul(Q,R)))
    
    # print(tabulate(V))
    print(VAL)
    
    print()
    # print(V)
    print(V1)
    