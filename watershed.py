import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "Dataset/349.jpg"
img = cv2.imread(path)

# resizing
img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)

# remove watermark
thresh = cv2.inRange(img, np.array([0, 0, 0]), np.array([128, 138, 135]))
scan = np.ones((3, 3), np.uint8)
cor = cv2.dilate(thresh, scan, iterations=1)
image_to_process = cv2.inpaint(img, cor, 5, flags=cv2.INPAINT_TELEA)
watermark_removed = image_to_process.copy()
# cv2.imshow("removed watermark", image_to_process)

# applying  AGT
# image_to_process = cv2.blur(
#     image_to_process, (3, 3))

gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)


kernel = np.ones((2, 2), np.uint8)

sure_bg = cv2.dilate(thresh, kernel, iterations=2)
thresh = 255 - thresh
# cv2.imshow("Threshold", thresh)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(thresh, 2, 3)
_, sure_fg = cv2.threshold(dist_transform, 0.8*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
# Find the markers for the watershed transform
num_components, markers = cv2.connectedComponents(sure_fg)

# Completing the markers
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(image_to_process, markers)
im = markers * gray
plt.imshow(im, 'gray')
plt.show()

# coloring boundaries
image_to_process[markers == -1] = [255, 0, 0]

# print("{} Bacteria colonies found".format(len(np.unique(markers)) - 1))

image = cv2.putText(image_to_process, "{} Bacterial colonies found".format(len(np.unique(markers)) - 1), (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.imshow('Overlay on original image', image)

# cv2.waitKey(0)

titles = ['Original Image', 'Watermark Removed',
          'Adaptive Gaussian Thresholding', 'Watershed Output']
images = [img, watermark_removed, thresh, image]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


# reference: https://github.com/bnsreenu/python_for_microscopists
