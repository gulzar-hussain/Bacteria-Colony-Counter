import cv2 
import numpy as np
from matplotlib import pyplot as plt 
import random

image = cv2.imread('Dataset/349.jpg')

image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#create historgram of image

#1 intensities
histo = np.zeros(256)


for j in range(image.shape[0]):
    for i in range(image.shape[1]):
        val = grayimage[j][i]
        #print(val)
        if(val>=0 and val<256):
           histo[val] = histo[val]+1
#plt.plot(histo)
#plt.show()

for j in range(image.shape[0]):
    for i in range(image.shape[1]):
        val = grayimage[j][i]
        if(val>=189 and val<=205):
           grayimage[j][i] = 0

cv2.imshow('image', grayimage)
cv2.waitKey()
cv2.destroyAllWindows()            




#2 based on hue
#histo = np.zeros(179)
#B, G, R = cv2.split(image)
#print(B[0][0])
#img2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
#print(len(h))
#print(len(h[0]))
#print(h[0][0])
#hist_h = cv2.calcHist([h],[0],None,[180],[0,180])
#plt.plot()
#plt.show()



