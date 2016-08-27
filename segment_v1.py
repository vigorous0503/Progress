import numpy as np
import cv2
import math
import pdb
import time
import copy
import numpy.matlib as npmat
from get_data import horizon
import scipy.io as sio
import skcuda.misc as misc
#import matplotlib.pyplot as plt
start=time.time()


#load segnet data
ID='289'
#inputname='outfile'+ID+'.npz'
inputname='outfile_crop'+ID+'.npz'
npzfile=np.load(inputname)
datacp=npzfile[npzfile.files[0]] #a copy for original data
#load the ground truth bounding box
bbox= './bbox/'+ID+'_track.txt'
#load the image
img= cv2.imread('crop_img'+ID+'.jpg')
#something shift to prevent log on negative number
min_data= np.amin(datacp,axis=2)
min_stack=np.repeat(min_data[:, :, np.newaxis], 12, axis=2)
data=datacp-min_stack+np.finfo(np.float32).eps#0.001

#segnet classes 0~11
"""
Sky
Building
Pole
Road Marking
Road
Pavement
Tree
Sign Symbol
Fence
Vehicle
Pedestrian
Bike
"""
#define union classes on layer representation
#Sky(L1):0 #Buiding(L2):1,2,6,7,8 #moving object(L3):9,10,11 #road(L4):3,4,5

beta=0.1 #param for energy function
#energy function with 6 channels: L1,L2,c9,c10,c11,L4
#pool the min energy in each union
energy=np.zeros((data.shape[0],data.shape[1],6))

datacost=(-1)*beta*np.log(data);
energy[:,:,0]=datacost[:,:,0];
energy[:,:,1]=np.amin(np.dstack((datacost[:,:,1],datacost[:,:,2],datacost[:,:,6],datacost[:,:,7],datacost[:,:,8])),axis=2)
energy[:,:,2:5]=datacost[:,:,9:12];
energy[:,:,5]=np.amin(np.dstack((datacost[:,:,4],datacost[:,:,3],datacost[:,:,5])),axis=2)
###

#integral image
Itg=np.cumsum(energy,axis=0)

bigN=np.iinfo('i').max#an infinite float pt num
###

#infence of layer representation
#h1 >=h2 >=h3

#minimize energy above h2 
E1=bigN*np.ones((energy.shape[0],energy.shape[0],energy.shape[1]))
E1[0,0,:]=0
for h2 in range(1,energy.shape[0]):
        for h3 in range(1,h2+1):
                E1[h2,h3,:]=Itg[h3-1,:,0]+Itg[h2-1,:,1]-Itg[h3-1,:,1]
             
Q=np.amin(E1,axis=1) #min energy function Q(h2)
Q_h3=np.argmin(E1,axis=1)  #record h3(h2) with min energy

# add the horizon contraint
end1=time.time()
v0= horizon(bbox,img)
end2=time.time()
print '---compute horizon line:  %s seconds --- ' % (end2 - end1)
print 'horizon line:', v0
#y_c= 1.60
yv_c= 1.50  	#vihicle car 3D height (m)
yv_b= 2.10  	#vihicle bus 3D height (m)
yp= 1.60  	#person 3D height (m)
yb= 1.20  	#bike 3D height (m)
yc= 1.40	#camera height
#v0= 240	 	#horizon line #240  1#200   2#120   5#160  8#250   9#235

vi= np.arange(0,data.shape[0])     	#object position on image 

#hv= y_c*abs(vi-v0)/yc
hv_c= yv_c*abs(vi-v0)/yc
hv_b= yv_b*abs(vi-v0)/yc
hp= yp*abs(vi-v0)/yc
hb= yb*abs(vi-v0)/yc
#hv[:v0]=bigN 		#It's impossible for the object position above the horizon line(v0)
hv_c[:v0]=bigN 		 
hv_b[:v0]=bigN
hp[:v0]=bigN
hb[:v0]=bigN

#create geometric cost map for object depth Qh(h1,h2,class)
#seperate 3 classes: vihecle, poeple, bike
#v_v=np.array(range(energy.shape[0]))
v_c_v=np.array(range(energy.shape[0]))
v_b_v=np.array(range(energy.shape[0]))
p_v=np.array(range(energy.shape[0]))
b_v=np.array(range(energy.shape[0]))

#best_v_u=v_v-hv
best_v_c_u=v_c_v-hv_c	#hope ideal h2 for every h1
best_v_b_u=v_b_v-hv_b
best_p_u=p_v-hp
best_b_u=b_v-hb
#best_v_u=np.repeat(best_v_u[:,np.newaxis], energy.shape[0], axis=1)
best_v_c_u=np.repeat(best_v_c_u[:,np.newaxis], energy.shape[0], axis=1)	#extend 
best_v_b_u=np.repeat(best_v_b_u[:,np.newaxis], energy.shape[0], axis=1)
best_p_u=np.repeat(best_p_u[:,np.newaxis], energy.shape[0], axis=1)
best_b_u=np.repeat(best_b_u[:,np.newaxis], energy.shape[0], axis=1)

hu=np.array(range(energy.shape[0]))	#all possible h2 according to every h1 
hu=np.repeat(hu[np.newaxis,:], energy.shape[0], axis=0)

#geo_cost_v=abs(best_v_u-hu)**2
geo_cost_v_c=abs(best_v_c_u-hu)**2	#define cost by distance(real h2,ideal h2)
geo_cost_v_b=abs(best_v_b_u-hu)**2
geo_cost_p=abs(best_p_u-hu)**2
geo_cost_b=abs(best_b_u-hu)**2

#np.fill_diagonal(geo_cost_v, 0) #if h1==h2: cost=0
np.fill_diagonal(geo_cost_v_c, 0) #if h1==h2: cost=0
np.fill_diagonal(geo_cost_v_b, 0) 
np.fill_diagonal(geo_cost_b, 0)
np.fill_diagonal(geo_cost_p, 0)

geo_cost_v=np.amin(np.dstack((geo_cost_v_c,geo_cost_v_b)),axis=2)

#minimize total energy
#seperate to car,people,bike 3 shannels  
E3v=bigN*np.ones((energy.shape[0],energy.shape[0],energy.shape[1]))
E3p=bigN*np.ones((energy.shape[0],energy.shape[0],energy.shape[1]))
E3b=bigN*np.ones((energy.shape[0],energy.shape[0],energy.shape[1]))

E3v[0,0,:]=Itg[energy.shape[0]-1,:,5]+Q[0,:]
E3p[0,0,:]=Itg[energy.shape[0]-1,:,5]+Q[0,:]
E3b[0,0,:]=Itg[energy.shape[0]-1,:,5]+Q[0,:]

k=10 #weight for geometric cost
for h1 in range(1,energy.shape[0]):
        for h2 in range(h1+1):
                
                
                E3v[h1,h2,:]=Itg[h1-1,:,2]-(h2!=0)*Itg[h2-1,:,2]+Itg[energy.shape[0]-1,:,5]-Itg[h1-1,:,5]+Q[h2,:]+k*geo_cost_v[h1,h2]
                E3p[h1,h2,:]=Itg[h1-1,:,3]-(h2!=0)*Itg[h2-1,:,3]+Itg[energy.shape[0]-1,:,5]-Itg[h1-1,:,5]+Q[h2,:]+k*geo_cost_p[h1,h2]
                E3b[h1,h2,:]=Itg[h1-1,:,4]-(h2!=0)*Itg[h2-1,:,4]+Itg[energy.shape[0]-1,:,5]-Itg[h1-1,:,5]+Q[h2,:]+k*geo_cost_b[h1,h2]

Qv=np.amin(E3v,axis=1)	#min energy function Qv(h1)
Qv_h2=np.argmin(E3v,axis=1)	#record h2_v(h1) with min energy
Qp=np.amin(E3p,axis=1)
Qp_h2=np.argmin(E3p,axis=1)
Qb=np.amin(E3b,axis=1)
Qb_h2=np.argmin(E3b,axis=1)
#find min energy in car,poeple,bike
M=np.dstack((Qv,Qp,Qb))
Q2=np.amin(M,axis=2)
Q2c=np.argmin(M,axis=2)

h1=np.argmin(Q2,axis=0) #h1 for every image column
L2=np.diag(Q2c[h1]) #get layer 2(moving obj) label
h2=(L2==0)*np.diag(Qv_h2[h1])+(L2==1)*np.diag(Qp_h2[h1])+(L2==2)*np.diag(Qb_h2[h1])
h3=np.diag(Q_h3[h2]) 

#columes talk
start_talk=time.time()
alpha=0.1
map2last_E=1000*np.ones((energy.shape[1],energy.shape[0]))
map2last_h=np.zeros((energy.shape[1],energy.shape[0]))

di=npmat.repmat(np.arange(energy.shape[0]),energy.shape[0],1)
dj=np.transpose(di)
degree=1
dij=abs(di-dj)^degree
#dij=1
#print dij
#print dij^2
#print dij
map2last_E[0,v0:energy.shape[0]-1]=0
for c in range(1,energy.shape[1]):        
		cross_cost=np.transpose(npmat.repmat(Q2[:,c],energy.shape[0],1))+npmat.repmat(Q2[:,c-1],energy.shape[0],1)
		cross_cost=cross_cost+npmat.repmat(map2last_E[c-1,:],energy.shape[0],1)+alpha*dij
                map2last_E[c,:]=np.amin(cross_cost,axis=1)
                map2last_h[c,:]=np.argmin(cross_cost,axis=1)

h1[energy.shape[1]-1]=np.argmin(map2last_E[energy.shape[1]-1,:])#head to back 
for i in range(energy.shape[1]-1,0,-1):
        h1[i-1]=map2last_h[i,h1[i]]
#print h1

end_talk=time.time()
print '%30s' % 'talk execution time ', str((end_talk - start_talk)*1000), 'ms'


#L2=np.diag(Q2c[h1]) #get layer 2(moving obj) label
#get h2, h3
#h2=(L2==0)*np.diag(Qv_h2[h1])+(L2==1)*np.diag(Qp_h2[h1])+(L2==2)*np.diag(Qb_h2[h1])



end3=time.time()
print '---inference : %s seconds --- ' % (end3 - end1)

#following is code for saving layer label for visaulization
smallTable=[1,2,6,7,8]
disp_seg=np.zeros((data.shape[0],data.shape[1]))

for j in range(data.shape[1]):
        disp_seg[:h3[j],j]=0;

        for i in range(h3[j],h2[j]):
                disp_seg[i,j]=smallTable[np.argmax([datacp[i,j,1],datacp[i,j,2],datacp[i,j,6],datacp[i,j,7],datacp[i,j,8]],axis=0)]             
        if(L2[j]==0): 
                disp_seg[h2[j]:h1[j],j]=9
        elif(L2[j]==1):
                disp_seg[h2[j]:h1[j],j]=10
        elif(L2[j]==2):
                disp_seg[h2[j]:h1[j],j]=11 
        for i in range(h1[j],data.shape[0]):
                disp_seg[i,j]=np.argmax(datacp[i,j,3:6],axis=0)+3

#np.savez('seg_crop_horizon'+ID,para=k,hori=v0,data=disp_seg) # layer+horizontal         
np.savez('seg_crop_horizon_talk'+ID,para=k,hori=v0,data=disp_seg,degree=degree,alpha=alpha)	# layer+horizontal+talk
#pdb.set_trace()
#draw result
disp_seg=np.zeros((energy.shape[0],energy.shape[1]))

for i in range(data.shape[1]):
	disp_seg[:h3[i],i]=0;
	disp_seg[h3[i]:h2[i],i]=1;
	if(L2[i]==0): 
		disp_seg[h2[i]:h1[i],i]=9;
	elif(L2[i]==1):
		disp_seg[h2[i]:h1[i],i]=10;
	elif(L2[i]==2): 
		disp_seg[h2[i]:h1[i],i]=11; 
	disp_seg[h1[i]:,i]=4;


disp_seg.astype(np.uint8)
#np.savez('layers'+ID,disp_seg)
#np.savez('layers_crop_horizon'+ID,disp_seg)
#np.savez('layers_crop_horizon_talk'+ID,disp_seg)
print('finish')
#end2=time.time()
#print '--- %s seconds --- ' % (end2 - start)

#cv2.imshow("segment",display2)
#cv2.waitKey(0)
