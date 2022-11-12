# import the opencv library
import cv2  
import numpy as np
from time import sleep

imgw, imgh = 1200, 900
amount = 20
brightness = 0.8
mimg = np.zeros(shape=[imgh, imgw, 3], dtype = "float32")

merges = 0

def reset():
    global mimg, merges

    mimg = np.zeros(shape=[imgh, imgw, 3], dtype = "float32")
    merges = 0

def merge(img):
    global mimg, merges

    img = cv2.resize(img, (imgw, imgh)) / 255

    factor = 1 / amount * brightness
    mimg = mimg + img * factor

    merges += 1

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(threshold, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

    greatest = 0
    approx = None

    for cnt in contours :
        area = cv2.contourArea(cnt)
    
        if area > 1000 and area < gray.size * 0.9:
            _approx = cv2.approxPolyDP(cnt, 
            0.009 * cv2.arcLength(cnt, True), True)
    
            if(len(_approx) == 4):
                if area > greatest:
                    approx = _approx
                    greatest = area
  
    if approx is not None:
        return four_point_transform(img,
        np.array([approx[0][0], approx[1][0],
        approx[2][0], approx[3][0]], dtype = "float32"))


vid = cv2.VideoCapture(0)
  
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if merges > amount:
        sleep(1)
        reset() 

    cv2.imshow('frame', mimg)
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    img = transform(frame)
  
    # Display the resulting frame
    if img is not None:
        merge(img)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()