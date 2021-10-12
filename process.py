import numpy as np
import cv2
import mediapipe as mp
import time
import math
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
cap.set(1000,800)
def find_angle(lm,img,p1,p2,p3,draw=True):
  p1 = tuple(lm[p1][1:])
  p2 = tuple(lm[p2][1:])
  p3 = tuple(lm[p3][1:])


  angle = math.degrees( math.atan2( (p3[1]-p2[1]),(p3[0]-p2[0]) )-math.atan2( (p2[1]-p1[1]), (p2[0]-p1[0]) )) 
  if angle<0:
    angle +=360
  angle2 = angle
  if angle2>115:
    angle2 = 170-(angle2-205)
    #angle2 = 105-(angle2-130)
  elif angle2<35:
    angle2 = 35
  global per
  global bar
  
  per = np.interp(angle2,(35,170),(0,100))#35,155
  bar = np.interp(angle2,(35,170),(320,30))#18,105
  if draw :
    cv2.line(img,p1,p2,(255,0,255),6)
    cv2.line(img,p2,p3,(255,0,255),6)
    
    cv2.circle(img,p1,11,(255,255,0),cv2.FILLED)
    cv2.circle(img,p1,15,(255,255,50),2)
    
    #cv2.putText(img,"Percentage "+str(int(per)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),4)
    #cv2.putText(img,str(int(per)),p2,cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),4)
    #cv2.putText(img,str(int(angle)),p2,cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),4)
    cv2.circle(img,p2,11,(255,255,0),cv2.FILLED)
    cv2.circle(img,p2,15,(255,255,50),2)
    
    cv2.circle(img,p3,11,(255,255,0),cv2.FILLED)
    cv2.circle(img,p3,15,(255,255,50),2)
per = 9
bar = 330
dir=0
# ptime = 0
start =False
c = 0
def process1(img):
	global per 
	global bar 
	global dir
	# global ptime 
	global start 
	global c 
 
 
	img = cv2.flip(img,1)
    
	keypoints = pose.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

	if keypoints.pose_landmarks:
		lm = []
		mp_drawing.draw_landmarks(img,keypoints.pose_landmarks,mp_pose.POSE_CONNECTIONS)
		for idx,landmark in enumerate(keypoints.pose_landmarks.landmark):
			lm.append([int(idx),int(landmark.x*img.shape[1]),int(landmark.y*img.shape[0])  ])
		
		#print(lm)	
		#find_angle(img,16,14,12)
		
		find_angle(lm,img,11,13,15)


		if len(lm)>0:
			start = True
		if per>=95.0 and start==True :
			if dir==1:
				c +=0.5
				dir=0
		if per<=35 and start==True:
			if dir==0:
				c +=0.5
				dir=1
	cv2.putText(img,"Count "+str(int(c)),(20,90),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
	cv2.rectangle(img, (570,30), (600, 330), (255, 200, 0), 3)
	color1 = (255, 200, 0)
	if per>=95 and start==True:
		color1 = (0, 200, 0)
	elif per<=35 and start==True:
		color1 = (0, 200, 0)
	
	cv2.rectangle(img, (570,int(bar)), (600, 330), color1, cv2.FILLED)

	# ctime = time.time()
	# fps = 1/(ctime-ptime)
	# ptime = ctime	
	cv2.putText(img,str(int(per)),(565,23),cv2.FONT_HERSHEY_PLAIN,2,color1,2)
	#cv2.putText(img,"FPS "+str(int(fps)),(20,40),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)
	# width = 1300
	# height = 670
	# dim = (width, height)

	# img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	return img





