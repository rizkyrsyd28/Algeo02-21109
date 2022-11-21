import os, shutil, cv2, natsort

def convertToGray(dir, output):
    # C:\tugas besar\ALGEO-2\data\color
    fileList = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            fileList.append(path)

    fileList = natsort.natsorted(fileList, key=lambda y: y.lower())

    for filename in os.listdir(output):
        file_path = os.path.join(output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    c = 1
    print(fileList)
    for filename in fileList:
        path1 = os.path.join(dir, filename)
        img = cv2.imread(path1, 0)
        img = cv2.resize(img, (256, 256))
        name = "Test" + str(c) + ".png"
        cv2.imwrite(output + name, img)
        c+=1;
    return c

def getGrayscale(dir):
    testImg = cv2.imread(dir, 0)
    testImg = cv2.resize(testImg, (256, 256))
    return testImg

if __name__ == "__main__":
    path = "D:\ITB\Semester 3\Aljabar Liniear dan Geometri\Algeo02-21109\data\\test\IMG-20221117-WA0003.jpg"
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite("austin.png", img)

# while(i <= N):
#     path = "../ALGEO-2/dataset/105_classes_pins_dataset/pins_Cristiano Ronaldo/Cristiano Ronaldo" + str(i) + ".jpg";
#     img = cv2.imread(path, 0);
#     img = cv2.resize(img, (256, 256));
    
#     name = "CR" + str(i) + ".png";
#     cv2.imwrite("../ALGEO-2/data/gray/" + name, img);

#     i += 1;
    

    

