import numpy as np
from averageface import avgFace
from getcovariant import getCov

def vecProjection (x, y) :
    return np.dot(x, y) / np.linalg.norm(y)

def qrFactorization(Cov, row, col) :

    U = np.empty(shape=(row, col), dtype=int)
    Q = np.empty(shape=(row, col), dtype=float)

    U[:,0] = Cov[:,0]
    Q[:,0] = U[:,0]/np.linalg.norm(U[:,0])

    for i in range (1,col) :
        UCol = U[:,i]
        CovCol = Cov[:,i]
        
        j = 0
        while j < i :
            uj = U[:,j]
            covi = Cov[:,i]
            subt = np.empty(shape=(row), dtype=float)
            subt += vecProjection(uj, covi)
            j += 1

        UCol = np.subtract(CovCol,subt)

        Q[:,i] = UCol/np.linalg.norm(UCol)

    return Q    

def getR (Q, Cov) :
    Qt = np.transpose(Q)
    return np.matmul(Qt,Cov)

def inputFromKeyBoard(row,col) :
    res = np.empty(shape=(row, col), dtype=int)

    for i in range (row) :
        for j in range (col) :
            res[i,j] = input()

    return res

if __name__ == "__main__":
    # Untuk testing masih ada hasil nan, 
    # mungkin bilangan kompleks atau pembagian dengan 0
    # res = np.empty(shape=(4, 4), dtype=int)
    # res = inputFromKeyBoard(4,4)
    # row = 4
    # col = 4

    # Q = qrFactorization(res, row, col)
    # R = getR(Q,res)
    # print(Q)
    # print(R)

    img = avgFace()
    covid = getCov(img)

    row = 98
    col = 98
    
    Q = qrFactorization(covid, row, col)
    R = getR(Q,covid)
    print(Q)
    print(R)
    