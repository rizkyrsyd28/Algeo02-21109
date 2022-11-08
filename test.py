import cv2
from numpy import linalg as al
import numpy as np

m = np.array([[3, -2, 4],[-2, 3, 1],[-1, 0, 5]], dtype=np.int8)

# _,eigenval, eigenvec = cv2.eigen(m)

# print(eigenval)
# print(_)

# V = al.eig(m)

# print(V[0])
# print(V[1])

qr= al.qr(m)

print("Q==============")
print(qr[0])
print("R==============")
print(qr[1])
print("M==============")
X = np.matmul(qr[0],qr[1])
X = X.astype(np.int8)
print(X)