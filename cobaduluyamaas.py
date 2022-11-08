import numpy as np
import getcovariant
import averageface
import math

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
    while (np.sum(np.square(res - init)) > margin):
        init = res
        q, r = qr(e)
        e = np.matmul(np.transpose(q), e)
        e = np.matmul(e, q)
        u = np.matmul(u, q)
        res = np.diagonal(e)
    res = np.around(np.diagonal(e), 6)
    vectors = u
    return (res, vectors)



if __name__ == "__main__":
    myimg = averageface.avgFace()
    curCov = getcovariant.getCovAlt(myimg)
    print(curCov)
    # contoh = np.array(
    #     [[26, 40, 41, 54],
    #     [40, 67, 62, 83],
    #     [41, 62, 95, 70],
    #     [54, 83, 70, 126]])
    q, r = qr(curCov)
    print("q:\n", q.round(6))
    print("r:\n", r.round(6))
    M = np.matmul(q, r)
    print("Multiplication result:\n", M.round(6))
    eVal, eVec = eigenValVec(curCov)
    print("\nMine")
    print(eVal)
    print(eVec)
    valveclist = np.linalg.eig(curCov)
    print("\nLibrary")
    print(valveclist[0])
    print(valveclist[1])
    # q, r = qr(curCov)