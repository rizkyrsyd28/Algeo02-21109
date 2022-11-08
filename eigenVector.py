import numpy as np
from qrFactorization import *

def eigenVector(Cov) :

    Q = qrFactorization(Cov, 98,98) 
    E = np.matmul(np.transpose(Q),Cov)
    E = np.matmul(E,Q)
    V = Q

    init = np.diag(Cov)
    res = np.diag(E)
    sum = np.sum(np.square(res-init))
    while (sum > 1e-15) :
        init = res

        Q = qrFactorization(E, 98,98) 
        E = np.matmul(np.transpose(Q),Cov)
        E = np.matmul(E,Q)
        V = np.matmul(V,Q)

        res = np.diag(E)
        sum = np.sum(np.square(res-init))

    return V

if __name__ == "__main__":
    img = avgFace()
    covid = getCov(img)
    V = np.empty(shape=(98, 98), dtype=int)

    row = 98
    col = 98
    
    V = eigenVector(covid)
    print(V)
    