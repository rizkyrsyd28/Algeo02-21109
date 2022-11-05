import numpy as np
import cv2
from averageface import avgFace

def getCov(avgimage):
    res = np.zeros([256, 256], dtype=int)

    for i in range(1, 99):
        path = "../ALGEO02-21109/data/gray/CR" + str(i) + ".png";
        im = cv2.imread(path, 0)
        temp = np.subtract(im, avgimage)
        transposed = np.transpose(temp)
        cov = np.matmul(temp, transposed)
        res = np.add(res, cov)

    res = np.multiply(1/98, res)
    return res

if __name__ == "__main__":
    img = avgFace()
    covid = getCov(img)
    print(covid)
    print(covid.shape)
