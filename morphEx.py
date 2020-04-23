import cv2
import numpy as np

IMG_PATH = "MorphTest.png"

def showWriteWait(name,img):
	cv2.imshow(name, img)
	cv2.imwrite(name +'.png',img)
	cv2.waitKey()
		
def main():
	image = cv2.imread(IMG_PATH) 
	cv2.imshow("Source image", image)
	cv2.waitKey()

	kernel = np.ones((3,3),np.uint8)
	
	erosion = cv2.erode(image,kernel,iterations = 1)
	showWriteWait("Erode",erosion)
		
	dilation = cv2.dilate(image,kernel,iterations = 1)
	showWriteWait("Dilatate",dilation)
	
	open_ = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	showWriteWait("Open",open_)
	

	close_ = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	showWriteWait("Close",close_)

	close_ = cv2.morphologyEx(open_, cv2.MORPH_CLOSE, kernel)
	showWriteWait("Open + Close",close_)
	

if __name__ == '__main__':
	main()