import cv2
import os

path = "../ALGEO-2/data/color/"
# C:\tugas besar\ALGEO-2\data\color
c = 1
for i in os.listdir(path):
    path1 = path + str(i);
    img = cv2.imread(path1, 0);
    img = cv2.resize(img, (256, 256));
    name = "Test" + str(c) + ".png";
    cv2.imwrite("../ALGEO-2/data/gray/" + name, img);
    c+=1;

# while(i <= N):
#     path = "../ALGEO-2/dataset/105_classes_pins_dataset/pins_Cristiano Ronaldo/Cristiano Ronaldo" + str(i) + ".jpg";
#     img = cv2.imread(path, 0);
#     img = cv2.resize(img, (256, 256));
    
#     name = "CR" + str(i) + ".png";
#     cv2.imwrite("../ALGEO-2/data/gray/" + name, img);

#     i += 1;
    

    

