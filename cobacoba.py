import numpy as np
import cobaduluyamaas

def addKZeros(arr, K):
    res = np.copy(arr)
    for i in range(K):
        res = np.append(res, 0)
    res = np.reshape(res, [1, res.size])
    return res


if (__name__ == "__main__"):
    testArr = np.array(
        [[3, -2, 0],
        [-2, 3, 0],
        [0, 0, 5]])
    eVal, eVec = cobaduluyamaas.eigenValVec(testArr, 1e-10000)
    print(eVal)
    print(eVec)
    test = np.linalg.eig(testArr)
    print(test[0])
    print(test[1])