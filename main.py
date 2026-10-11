import cv2 as cv
import sys
import numpy as np

img = cv.imread("/home/sebastian/uni/year3/Machine_Vision/Course-Project/Data_Sets/Data_Set_2/dataset2-master/dataset2-master/images/test/NEUTROPHIL/_43_289.jpeg")
if img is None:
    sys.exit("Could not load image")

#make grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#otsu binarization
#also the binary trashholding inversed
ret, tresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#noise removal
kernel = np.ones((3,3), np.uint8)
#morphological opening removes small holes in object per tutorial, but this doesnt rlly have any holes?
opening = cv.morphologyEx(tresh, cv.MORPH_OPEN, kernel, iterations=2)

#sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

#finding sure forground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, float(0.7*dist_transform.max()), 255, 0)

#finding unknown region
sure_fg = sure_fg.astype(np.uint8)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]


while 1:

    cv.imshow("window", img)
    k = cv.waitKey(0)
    if k == ord("q"):
        cv.destroyAllWindows()
        break

