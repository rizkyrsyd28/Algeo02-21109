import numpy as np
import cv2
import os

def avgFace():
    init = np.zeros([256, 256], dtype=int)
    for i in range(1,51):
        path = "../ALGEO02-21109/data/gray/Test" + str(i) + ".png";
        img = cv2.imread(path, 0)
        init = np.add(init, img)
    
    init = np.multiply(1/98, init)
    init = init.astype(np.uint8)
    return init

def getAvgFace(dir):
    init = np.zeros([256, 256], dtype=int)
    count = 0
    for filename in os.scandir(dir):
        if filename.is_file():
            img = cv2.imread(filename.path, 0)
            init = np.add(init, img)
            count += 1
    init = init / count
    init = init.astype(np.uint8)
    return init

# img = cv2.imread("../ALGEO-2/data/gray/CR1.png", 0);
if __name__ == "__main__":
    im = getAvgFace("D:\ITB\Semester 3\Aljabar Liniear dan Geometri\Algeo02-21109\data\\newgray");
    cv2.imshow("cobaaja", im);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
    im = avgFace();
    cv2.imshow("cobaaja", im);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
