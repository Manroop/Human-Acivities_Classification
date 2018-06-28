import os
import numpy as np
import cv2

newpath = r'/home/mst/Desktop/HAC/Images/Radar_Images_png/Test_Images/walkinglow/'
path, dirs, files = next(os.walk(newpath))
print (files[32])
file_count = len(files)
print (file_count)

for i in range (file_count):
    print (i)
    if i != 16:
        img = cv2.imread(files[i])
        cropped_img = img[200:420, 0:140]
        s = "walkinglow_"+str(i)+".jpg"
        cv2.imwrite(s, cropped_img)
