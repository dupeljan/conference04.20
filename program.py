import cv2
import numpy as np

IMG_PATH = "text.png"

def main():
	image = cv2.imread(IMG_PATH) 
	cv2.imshow("Source image", image)
	cv2.waitKey()

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image,(5,5),0)
	image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)[1]
	cv2.imshow("Gray image", image)
	cv2.waitKey()

	leftBorder = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
	rightBorder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32)
	leftBorderR = cv2.filter2D(image,-1,leftBorder)
	rightBorderR = cv2.filter2D(image,-1,rightBorder)
	image = cv2.addWeighted(leftBorderR,1,rightBorderR,1,0)
	cv2.imshow("Gradient image", image)
	cv2.waitKey()

	'''
	kernel = np.ones((10,1),np.float32)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	'''
	kernel = np.ones((10,1),np.float32),np.ones((32,1),np.float32)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[0])
	mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel[1])
	image = cv2.bitwise_not(image)
	image = cv2.addWeighted(image,1,mask,1,0)

	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5,15),np.float32),iterations=2)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5,15),np.float32))

	cv2.imshow("Closing", image)
	cv2.waitKey()
	
if __name__ == '__main__':
	main()