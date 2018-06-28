import scipy.io as spio
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import os

'''
p = ['Human_Spect_test9_boxingmoving_04.mat','Human_Spect_test9_boxingmoving_03.mat']

#f = open("image_text.png","w+")
for i in range (1):
    mat = spio.loadmat(p[i])
    Image = mat['TimeFreq_f']
    s = "boxingmoving_0"+str(i)+".png"
    plt.imsave(s, Image)
'''
newpath = r'/home/mst/Desktop/HAC/Images/Radar_Images/boxingmoving/'
path, dirs, files = next(os.walk(newpath))
print (files[0])
file_count = len(files)
print (file_count)

for i in range (143):
    mat = spio.loadmat(files[i])
    Image = mat['TimeFreq_f']
    s = "boxingmoving_0"+str(i)+".png"
    plt.imsave(s, Image)
