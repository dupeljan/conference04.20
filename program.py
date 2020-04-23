import cv2
import numpy as np

IMG_PATH = "testImage1.png"

def main():
	image = cv2.imread(IMG_PATH) 
	cv2.imshow("Source image", image)
	cv2.waitKey()

	# Prepare image to
	# find borders
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image,(5,5),0)
	image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)[1]
	name = "Prepaired image"
	cv2.imshow(name, image)
	cv2.imwrite(name +'.png',image)
	cv2.waitKey()

	# Compute image gradient
	leftBorder = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
	rightBorder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
	leftBorderR = cv2.filter2D(image,-1,leftBorder)
	rightBorderR = cv2.filter2D(image,-1,rightBorder)
	image = cv2.addWeighted(leftBorderR,1,rightBorderR,1,0)
	name = "Gradient image"
	cv2.imshow(name, image)
	cv2.imwrite(name +'.png',image)
	cv2.waitKey()

	# Do morphologic operation
	# to filter text blocks
	kernel = np.ones((8,1),np.float32),np.ones((32,1),np.float32)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[0])
	mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[1])
	cv2.imshow("Mask", mask)
	cv2.imwrite(name +'.png',image)
	cv2.waitKey()
	
	image = cv2.bitwise_not(image)
	image = cv2.addWeighted(image,1,mask,1,0)
	
	# Image inversed now
	# CLOSE and OPEN become inversed too 
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5,15),np.float32),iterations=2)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5,15),np.float32))

	name = "Result"
	cv2.imshow(name, image)
	cv2.imwrite(name +'.png',image)
	cv2.waitKey()
	

	
if __name__ == '__main__':
	main()