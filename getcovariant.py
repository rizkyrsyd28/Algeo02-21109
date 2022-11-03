import numpy as np
import cv2
from averageface import avgFace

def getCov(avgimage):
    res = np.empty(shape=(65536, 0), dtype=int)

    for i in range(1, 99):
        path = "../ALGEO02-21109/data/gray/CR" + str(i) + ".png";
        im = cv2.imread(path, 0)
        temp = np.subtract(im, avgimage)
        temp = temp.reshape([65536, 1])
        res = np.append(res, temp, axis=1)

    transposed = np.transpose(res)
    cov = np.matmul(transposed, res)

    return cov

if __name__ == "__main__":
    img = avgFace()
    covid = getCov(img)
    print(covid)

