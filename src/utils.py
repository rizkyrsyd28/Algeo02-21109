import os
import cv2
import numpy as np

def temp_folder():
    eigen = "parahlimpik/eigen/"

    currentPath = os.getcwd()
    print(currentPath)
    path = os.path.join(currentPath, eigen) 
    os.mkdir(path)

def crop_cam(mat):
    height, width, _ = mat.shape
    center = width//2
    mat = mat[0:height, center - height//2: center + height//2]
    return mat