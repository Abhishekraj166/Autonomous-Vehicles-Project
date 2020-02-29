import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression

import read
mainFolder= "C:/Users/akans/Desktop/a/FullIJCNN2013//"
dataset, file_list, annotation = read.read(mainFolder)
img = cv2.imread(mainFolder + file_list[11])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

template = dataset [79][1]
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

w,h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCDEFF_NORMED)
threshold = 0.8
loc = np.where (res>=threshold)
results=[]
for pt in zip(*loc[::-1]):
    results.append([pt[0]+w, pt[1]+h])