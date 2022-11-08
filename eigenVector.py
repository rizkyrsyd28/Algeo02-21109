import numpy as np
from qrFactorization import *

def eigenVector(Cov) :

    Q = qrFactorization(Cov, 256,256) 
    E = np.matmul(np.transpose(Q),Cov)
    E = np.matmul(E,Q)
    V = Q

    init = np.diag(Cov)
    res = np.diag(E)
    sum = np.sum(np.square(res-init))
    while (sum > 1e-15) :
        init = res

        Q = qrFactorization(E, 256,256) 
        E = np.matmul(np.transpose(Q),Cov)
        E = np.matmul(E,Q)
        V = np.matmul(V,Q)

        res = np.diag(E)
        sum = np.sum(np.square(res-init))

    return V

if __name__ == "__main__":
    img = avgFace()
    covid = getCov(img)
    V = np.empty(shape=(256, 256), dtype=int)

    # row = 256
    # col = 256
    
    V = eigenVector(covid)
    print(V)
    