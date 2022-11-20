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

    for i in range(500000) :
        shift = np.multiply(CovCopy.item(row-1,col-1),I)
        CovCopy = np.subtract(CovCopy,shift)
        Q = qrFactorization(CovCopy, row, col)
        R = getR(Q, CovCopy)
        # Q, R = np.linalg.qr(CovCopy)
        CovCopy = np.matmul(R,Q)
        CovCopy = np.add(CovCopy,shift)
        Qn = np.matmul(Qn,Q)

    # print(Qn)
    eigenVal = np.diag(Qn)
    eigenVec = np.empty(shape=(row,col), dtype = float)
    zeros = np.zeros(shape=(0,col), dtype = int)
    # eigenVal = np.sort(eigenVal)[::-1]
    # eigenVal = np.reshape(eigenVal,[4,1])
    return eigenVal
    # print(eigenVal.shape)
    
    # shp, NULL =eigenVal.shape

    # for i in range(shp) :
    #     lambdaI = np.multiply(eigenVal[i,0],I)
    #     print(np.linalg.solve(np.subtract(lambdaI,Cov),zeros).shape)
    #     lambdaI = np.linalg.solve(np.subtract(lambdaI,Cov),zeros)
    #     eigenVector[:,i] = np.reshape(lambdaI, [4,])
        
    # return eigenVector

def eigen_qr_practical(A, iterations=500000):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        # s_k is the last item of the first diagonal
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        # pe perform qr and subtract smult
        Q, R = np.linalg.qr(np.subtract(Ak, smult))
        # we add smult back in
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q
        # if k % 10000 == 0:
        #     print("A",k,"=")
        #     print(tabulate(Ak))
        #     print("\n")
    return Ak, QQ

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
    VAL1, V1 = np.linalg.eig(res)
    VAL2, V2 = eigen_qr_practical(res)

    V3 = eigenVectorNew(res)
    
    # print(tabulate(V))
    print(VAL1, "\n")
    print("Batas val1", "\n")
    
    print(VAL2,"PPP\n")
    print(V3)
    # print(V)
    print(tabulate(V1))
    print(V2)
    