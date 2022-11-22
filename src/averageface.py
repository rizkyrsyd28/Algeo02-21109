import numpy as np
import cv2
import os

def getAvgFace(dir):
    init = np.zeros([256, 256], dtype=int)
    count = 0
    for filename in os.scandir(dir):
        if filename.is_file():
            img = cv2.imread(filename.path, 0)
            img = cv2.resize(img, (256, 256))
            init = np.add(init, img)
            count += 1
    init = np.multiply(1/count, init)
    init = init.astype(np.uint8)
    return init, count

# img = cv2.imread("../ALGEO-2/data/gray/CR1.png", 0);
if __name__ == "__main__":
    im = getAvgFace(".\data\gray");
    cv2.imshow("cobaaja", im);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
