import os
import cv2
import numpy as np

def temp_folder():
    eigen = "/eigen_parahlimpik"

    currentPath = os.getcwd()
    print(currentPath)
    path = os.path.join(currentPath[2:] + eigen) 
    if (not os.path.exists(currentPath[2:] + eigen)):
        os.mkdir(path)
        print("[CREATE] : C:/" +currentPath[2:] + eigen)
    else :
        print("[WARN] : FILE ALREADY EXIST")

def crop_cam(mat):
    height, width, _ = mat.shape
    center = width//2
    mat = mat[0:height, center - height//2: center + height//2]
    return mat