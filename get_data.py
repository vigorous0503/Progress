import numpy as np
import os 
import cv2
import pdb
#import horizon_calculated
def horizon(bbox, img):

	f = open(bbox)
	#img=cv2.imread('./usedframe/150_2.jpg')
	#v = open('tracking_gt','w')
	text=f.read()
	text=text.split('\n')
	Nlines=len(text)-1

	camera_height=1.40
	HEIGHT = {'bike':1.05, 
		'bus':2.10, 
		'car':1.50, 
		'motorbike':1.20, #1.60  #1.20
		'person':1.00
		}

	gt = [[] for x in range(Nlines)]
	#print len(gt)


	for i in range(Nlines):
		gt[i]=text[i].split(' ')

	#print(gt[10])
	frame_count=1
	horizon_line=np.zeros(100)
	boxes=np.zeros(100)
	for i in range(Nlines):
		frame_ind=int(gt[i][0])-1 #frame index
		gt[i][2]=gt[i][2].split('"')[1].split('"')[0] #class 
		a_line=int(gt[i][6])-(int(gt[i][6])-int(gt[i][4]))*camera_height/float(HEIGHT[gt[i][2]])
		horizon_line[frame_ind]+=a_line #accumulate horizons
		boxes[frame_ind]+=1 #number of horizons
	
	horizon_line=horizon_line/boxes #average over number of boxes
	#print(horizon_line[0]) #horizon of first frame

	#draw the horizon line on the crop image 
	img1= np.zeros((img.shape[0],img.shape[1]))
	cv2.line(img1,(0,int(horizon_line[0])),(img.shape[1]-1,int(horizon_line[0])),255)
	img1= cv2.resize(img1,(480,360))
	horizon= np.where(img1>0)[0][0]

	#frame = cv2.resize(img, (480,360))
	#cv2.imwrite('re_img'+ID+'.jpg', frame)
	#temp1= frame    #draw hoirizon line on original(360*480) image
	#cv2.line(temp1,(0,horizon),(temp1.shape[1]-1,horizon),(0,0,255))
        #cv2.imwrite( './results/ori_crop'+ID+'.jpg',temp1 );
	#cv2.line(img,(0,int(horizon_line[0])),(img.shape[1]-1,int(horizon_line[0])),(255,0,0))
	#cv2.imwrite(''./results/img'+ID+'.jpg',img)
	#cv2.imshow('photo',img)
	#cv2.waitKey(0)
	f.close()
	return horizon
