import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
from imutils.object_detection import non_max_suppression

def read (mainFolder):
    xx = os.listdir(mainFolder)

    folder_list =[]
    file_list = []

    for name in xx:
        if ".ppm" in name:
            file_list.append(name)
        elif len(name)==2:
            folder_list.append(name)
        
    dataset= []

    for folder in folder_list:
        strTemp = mainFolder + folder
        xx=os.listdir(strTemp)
        
        count = 0
        for name in xx:
            img = plt.imread(strTemp + "//" + name)
            dataset.append([strTemp + "//" +  name, img])
            count = count+1
            if count > 10:
                break
        annotation = {}

    with open(mainFolder+"gt.txt","r") as inst:
        for line in inst:
            filename,x1,y1,x2,y2,t = line.split(";")
            
            if filename in annotation:
                annotation[filename].append([int(x1),int(y1),int(x2),int(y2)])
            else:
                annotation[filename] = [int(x1),int(y1),int(x2),int(y2)]
                
    return dataset, file_list, annotation

mainFolder = "C:/Users/akans/Desktop/a/FullIJCNN2013//"
dataset, file_list, annotation = read(mainFolder)
img = cv2.imread(mainFolder + file_list[11])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = dataset [79][1]
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.4

  
loc = np.where(res>=threshold)

results=[]
for pt in zip(*loc[::-1]):
    results.append([pt[0] ,pt[1] ,pt[0]+w,pt[1] + h])
    
for x1,y1,x2,y2 in results:
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
plt.imshow(img)
