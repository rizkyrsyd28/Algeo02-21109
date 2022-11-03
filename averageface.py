import numpy as np
import cv2

def avgFace():
    init = np.zeros([256, 256], dtype=int);
    for i in range(1,99):
        path = "../ALGEO02-21109/data/gray/CR" + str(i) + ".png";
        img = cv2.imread(path, 0);
        init = np.add(init, img);
    
    init = np.multiply(1/99, init);
    init = init.astype(np.uint8);
    return init;

# img = cv2.imread("../ALGEO-2/data/gray/CR1.png", 0);
if __name__ == "__main__":
    im = avgFace();
    cv2.imshow("sewey", im);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
