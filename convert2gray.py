import cv2

i = 1;
N = 98;

while(i <= N):
    path = "../ALGEO-2/dataset/105_classes_pins_dataset/pins_Cristiano Ronaldo/Cristiano Ronaldo" + str(i) + ".jpg";
    img = cv2.imread(path, 0);
    img = cv2.resize(img, (256, 256));
    
    name = "CR" + str(i) + ".png";
    cv2.imwrite("../ALGEO-2/data/gray/" + name, img);

    i += 1;
    

    

