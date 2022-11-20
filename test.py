import cv2

cap = cv2.VideoCapture(1)

img = cap.read()[1]
print(img)

cap.release()

print(img)
print(img)
print(img)
print(img)