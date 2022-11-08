import numpy as np
from averageface import avgFace
from getcovariant import getCov
from tabulate import tabulate

def vecProjection (x, y) :
    return np.multiply((np.dot(x,y)/np.dot(y,y)),y)

def qrFactorization(Cov, row, col) :

    U = np.empty(shape=(row, col), dtype=float)
    Q = np.empty(shape=(row, col), dtype=float)

    U[:,0] = -Cov[:,0]
    Q[:,0] = np.multiply(U[:,0],1/np.linalg.norm(U[:,0]))

    for i in range (1,col) :
        UCol = U[:,i]
        CovCol = Cov[:,i]
        
        j = 0
        subt = np.zeros(shape=(row), dtype=float)
        
        while j < i :
            uj = U[:,j]
            covi = Cov[:,i]
            subt = np.add(subt,vecProjection(covi, uj))
            j += 1

        UCol = np.subtract(CovCol,subt)
        U[:,i] = UCol

        if (np.linalg.norm(UCol) == 0) : 
            print("Matrix has linearly dependent columns")
            return None

        Q[:,i] = np.multiply(UCol,1/np.linalg.norm(UCol))

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
    res = np.empty(shape=(4, 4), dtype=int)
    res = inputFromKeyBoard(4,4)
    row = 4
    col = 4

    Q = qrFactorization(res, row, col)
    R = getR(Q,res)

    Q1, R1 = np.linalg.qr(res)

    print(tabulate(Q))
    print(tabulate(Q1))

    # print(tabulate(R))
    # print(tabulate(R1))
    


    # img = avgFace()
    # covid = getCov(img)

    # row = 256
    # col = 256
    
    # Q1 = qrFactorization(covid, row, col)
    # R = getR(Q1,covid)
    # print(Q1)
    # print(R)
    
    # Q, R = np.linalg.qr(covid)
    # print(tabulate(R))

    # x1, y1 = Q1.shape
    # x, y = Q.shape
    
    # for i in range(x) :
    #     for j in range(y):
    #         if (Q1[i][j] != Q[i][j]) :
    #             print(Q1[i][j], Q[i][j])
                
    #             break
        
