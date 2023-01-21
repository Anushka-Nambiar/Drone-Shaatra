import cv2
import numpy as np

VideoSize=[None,None]
Center=[None,None]
approx=50

class Cam():
	def __init__(self,name,camID):
		self.name=name
		self.camID=camID
	def run(self):
		self.camView()
	def camView(self):
		global Center,VideoSize
		video=cv2.VideoCapture(self.camID)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('output3.avi', fourcc, 20.0, (640, 480))
		while(True):
			ret,image=video.read()
			(VideoSize[0],VideoSize[1])=(x,y)=(len(image[0]),len(image))
			marked=colorDetector(image)
			marked=arucoDetector(image)
			cv2.circle(marked,(x//2,y//2),3,(0,0,255),1)
			cv2.line(marked,(x//2,y//2),(0,0),(255,0,255),1)
			cv2.line(marked,(x//2,y//2),(len(marked[0]),0),(255,0,255),1)
			cv2.line(marked,(x//2,y//2),(0,len(marked)),(255,0,255),1)
			cv2.line(marked,(x//2,y//2),(len(marked[0]),len(marked)),(255,0,255),1)
			cv2.rectangle(marked,(x//2-approx,y//2-approx),(x//2+approx,y//2+approx),(255,0,0),1)
			if(Center[0] is not None):
				if(abs(Center[0])<approx and abs(Center[1])<approx):
					cv2.putText(image,"Landing",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
				else:
					if(Center[0]<=Center[1] and -Center[0]<=Center[1]):
						cv2.putText(image,"Backwards",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
					elif(-Center[0]>=Center[1] and Center[0]<=Center[1]):
						cv2.putText(image,"Left",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
                    			elif(Center[0]>=Center[1] and -Center[0]<=Center[1]):
                        			cv2.putText(image,"Right",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
                    			elif(Center[0]>=Center[1] and -Center[0]>=Center[1]):
                        			cv2.putText(image,"Forward",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
            		else:
                		cv2.putText(image,"Idle",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.75, (0,0,0), 2)
			out.write(marked)
            		cv2.imshow(self.name,marked)
            		if cv2.waitKey(20)==27:
                		break
        		video.release()
       			cv2.destroyWindow(self.name)
def colorDetector(image):
	global Center
   # img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   # lower = np.array([15, 150, 20])
   # upper = np.array([35, 255, 255])
   # mask = cv2.inRange(img, lower, upper)
   # mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # if len(mask_contours) != 0:
     #   for mask_contour in mask_contours:
          #  if cv2.contourArea(mask_contour) > 500:
         #       x, y, w, h = cv2.boundingRect(mask_contour)
         #       cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 3)
        #        Center[0]=int(((x+w)-VideoSize[0]/2)/2)
       #         Center[1]=int(((y+w)-VideoSize[1]/2)/2)
      #          cX=int(x+w/2)
     #           cY=int(x+h/2)
    #            cv2.circle(image,(cX,cY),1,(255,0,0),-1)
   # else:
    #    Center=[None,None]
    #return image
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    
	green_lower = np.rray([25, 52, 72], np.uint8)
	green_upper = np.array([102, 255, 255], np.uint8)
	green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    	kernal = np.ones((5, 5), "uint8")

    	green_mask = cv2.dilate(green_mask, kernal)
    	res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)

    	contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x,y,w,h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),(x+w, y+h), (0, 255, 0), 2)
			cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    	cv2.imshow("Green Detected", imageFrame)

def arucoDetector(image):
    	global Center
   	aruco=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
    	arucoParams = cv2.aruco.DetectorParameters_create()
    	corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco, parameters=arucoParams)
   	if(len(corners)>0):
		ids = ids.flatten()
		for (markerCorner, markerID) in zip(corners, ids):
			corners = markerCorner.reshape((4, 2))
            		(topLeft, topRight, bottomRight, bottomLeft) = corners
            		topRight = (int(topRight[0]), int(topRight[1]))
            		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            		topLeft = (int(topLeft[0]), int(topLeft[1]))
            		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            		Center[0]= int((topLeft[0] + bottomRight[0])/2-VideoSize[0]/2)
            		Center[1]=int((topLeft[1] + bottomRight[1])/2-VideoSize[1]/2)
            		cX= int((topLeft[0] + bottomRight[0]) / 2.0)
            		cY=int((topLeft[1] + bottomRight[1]) / 2.0)
            		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            		if(markerID==1):
                		cv2.putText(image, str(markerID)+" DropZone",(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
            		elif(markerID==2):
                		cv2.putText(image, str(markerID)+" LandingZone",(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
            		else:
                		cv2.putText(image, str(markerID)+" Package",(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0,0), 2)
    		return image

if __name__=="__main__":
    cam=Cam("Frontcam",0)
    cam.run()

