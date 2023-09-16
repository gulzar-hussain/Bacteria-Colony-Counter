import cv2 
import numpy as np
import matplotlib
import glob


# images = [cv2.imread(file) for file in glob.glob("Dataset/dataset/*.jpg")]


# for image in images:
#     smallimage = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
#     #smallimage = cv2.cvtColor(smallimage, cv2.COLOR_BGR2GRAY)
#     #smallimage, thresh = cv2.threshold(smallimage,128,255,cv2.THRESH_BINARY)

#     #remove watermark 
#     thresh = cv2.inRange ( smallimage , np.array([0,0,0]) , np.array ( [ 128,138,135 ] ) ) 
#     scan= np.ones((3,3),np.uint8) 
#     cor = cv2.dilate(thresh,scan, iterations=1) 
#     specular = cv2.inpaint( smallimage , cor , 5 , flags = cv2.INPAINT_TELEA )

#     # cv2.imshow("removed watermark", specular)

#     # ref: https://blog.katastros.com/a?ID=00850-c4b6a7b5-dd9b-4707-a0a8-19cb70f84c4c


#     #blur to remove noise

#     #average blur
#     ablur = cv2.blur(specular,(3,3))
#     # cv2.imshow("average blur",ablur)
#     #gaussian blur
#     gblur = cv2.GaussianBlur(specular,(3,3),cv2.BORDER_DEFAULT) 
#     # cv2.imshow("gaussian blur",gblur)
#     # median blur
#     mblur = cv2.medianBlur(specular,7)
#     # cv2.imshow("median blur",blur)

#     #edge detection 
#     # border colour = rgb = 97 , 112 , 79
#     # alge rgb = 156, 162 , 92
#     #circle out petri dish 
#     #maybe can use erode to remove
#     # fill colour using inpaint 
#     # thresh2 = cv2.inRange ( specular , np.array([97,112,79]) , np.array ( [ 156,162,92 ] ) ) 
#     # scan2= np.ones((3,3),np.uint8) 
#     # cor2 = cv2.dilate(thresh2,scan2, iterations=1) 
#     # specular2 = cv2.inpaint( specular , cor2 , 5 , flags = cv2.INPAINT_TELEA )
#     # cv2.imshow("remove petri dish", specular2)

#     #edges present in image 
#     # canny = cv2.Canny(gblur,125,175)
#     canny = cv2.Canny(ablur,125,175)
#     # canny = cv2.Canny(mblur,125,175)
#     # canny = cv2.Canny(specular,125,175)
#     # canny = cv2.Canny(specular,50,150)
#     #canny = cv2.Canny(gblur,25,100)
#     cv2.imshow("image", canny)
#     #remove bad contours
    

#     contours , hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) 
#     # contours , hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
#     print("Number of bacterial colonies= ", len(contours))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


image = cv2.imread('349.jpg')
# image = cv2.imread('356.jpg')
smallimage = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
thresh = cv2.inRange ( smallimage , np.array([0,0,0]) , np.array ( [ 128,138,135 ] ) ) 
scan= np.ones((3,3),np.uint8) 
cor = cv2.dilate(thresh,scan, iterations=1) 
specular = cv2.inpaint( smallimage , cor , 5 , flags = cv2.INPAINT_TELEA )


ablur = cv2.blur(specular,(3,3))
# cv2.imshow("average blur",ablur)
#gaussian blur
gblur = cv2.GaussianBlur(specular,(3,3),cv2.BORDER_DEFAULT) 
# cv2.imshow("gaussian blur",gblur)
# median blur
mblur = cv2.medianBlur(specular,7)
# cv2.imshow("median blur",blur)

canny = cv2.Canny(ablur,125,175)
# canny = cv2.Canny(gblur,125,175)
# canny = cv2.Canny(mblur,125,175)
cv2.imshow("image", canny)
contours , hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) 
# contours , hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
print("Number of bacterial colonies= ", len(contours))
