import numpy as np
import cv2
from eigenVector import *

def getDifferences(im,avgimage):
    res = np.empty(shape=(98, 98), dtype=int)

    for i in range(1, 99):
        temp = np.subtract(im, avgimage)
        res = np.append(res, temp, axis=1)

    transposed = np.transpose(res)
    cov = np.matmul(transposed, res)

    return cov

def eigenFaces(V, avgimage) :

    eFace = np.empty(shape=(98, 98, 98), dtype=float)

    for i in range(1, 98):
        path = "../ALGEO02-21109/data/gray/CR" + str(i) + ".png";
        im = cv2.imread(path, 0)
        sub = np.subtract(im, avgimage)
        res = np.matmul(V,sub)
        eFace = np.append(eFace, res, axis=1)

    return eFace