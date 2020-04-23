import cv2
import numpy as np
import os

IMG_PATH = "imageSource/testImage1.png"
OUT_DIR = "programOut"

def showWriteWait(name,img):
	cv2.imshow(name, img)
	cv2.imwrite(os.path.join(OUT_DIR,name +'.png'),img)
	cv2.waitKey()

def main():
	# Create dir for pictures
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	# Read image
	image = cv2.imread(IMG_PATH) 
	cv2.imshow("Source image", image)
	cv2.waitKey()

	# Prepare image to
	# find borders
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image,(5,5),0)
	image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)[1]
	showWriteWait("Prepaired image",image)

	# Compute image gradient
	leftBorder = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
	rightBorder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
	leftBorderR = cv2.filter2D(image,-1,leftBorder)
	rightBorderR = cv2.filter2D(image,-1,rightBorder)
	image = cv2.addWeighted(leftBorderR,1,rightBorderR,1,0)
	showWriteWait("Gradient image",image)

	# Do morphologic operation
	# to filter text blocks
	kernel = np.ones((8,1),np.float32),np.ones((32,1),np.float32)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[0])
	mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[1])
	
	image = cv2.bitwise_not(image)
	image = cv2.addWeighted(image,1,mask,1,0)
	
	# Image inversed now
	# CLOSE and OPEN become inversed too 
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5,15),np.float32),iterations=2)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5,15),np.float32))

	showWriteWait("Result",image)

	# Draw contours on image
	contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	image = cv2.imread(IMG_PATH) 
	color = (0,255,0) # Green
	# if you dont want bound rect, uncoment
	#cv2.drawContours(image, contours, -1, (255,0,0), 1)
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		image = cv2.rectangle(image,(x,y),(x+w,y+h),color,1)
	
	showWriteWait("Contours",image)

	
if __name__ == '__main__':
	main()